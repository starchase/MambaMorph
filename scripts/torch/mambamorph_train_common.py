#!/usr/bin/env python3
"""Shared utilities for unsupervised diffeomorphic MambaMorph training."""

import csv
import random
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(inshape, device, resume=None):
    # TransMorph.py uses historical top-level imports (layers, networks, mamba).
    # Match the repository's original train_cross.py import convention.
    repo_root = Path(__file__).resolve().parents[2]
    torch_module_dir = repo_root / 'mambamorph' / 'torch'
    if str(torch_module_dir) not in sys.path:
        sys.path.insert(0, str(torch_module_dir))
    # TransMorph imports VoxelMorph-only classes at module import time. They are
    # not used by MambaMorph, and importing the full networks module also pulls
    # in optional flash-attn. Keep this baseline independent of that extension.
    if 'networks' not in sys.modules:
        networks_stub = types.ModuleType('networks')

        class UnusedVoxelMorphModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError('This training entrypoint only supports MambaMorph.')

        networks_stub.Unet = UnusedVoxelMorphModule
        networks_stub.ConvBlock = UnusedVoxelMorphModule
        sys.modules['networks'] = networks_stub
    from TransMorph import CONFIGS, MambaMorph

    if any(size % 16 for size in inshape):
        raise ValueError(f'MambaMorph requires every input dimension to be divisible by 16, got {inshape}.')
    config = CONFIGS['MambaMorph']
    config.img_size = tuple(inshape)
    model = MambaMorph(config).to(device)
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        print(f'Loaded checkpoint: {resume}')
    return model


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


class LocalNCC(torch.nn.Module):
    def __init__(self, win=9, eps=1e-3):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, fixed, moved, mask=None):
        filt = torch.ones((1, 1, self.win, self.win, self.win), device=fixed.device, dtype=fixed.dtype)
        padding = self.win // 2
        conv = lambda x: F.conv3d(x, filt, padding=padding)
        count = float(self.win ** 3)
        fixed_sum, moved_sum = conv(fixed), conv(moved)
        fixed2_sum, moved2_sum = conv(fixed * fixed), conv(moved * moved)
        cross_sum = conv(fixed * moved)
        fixed_mean, moved_mean = fixed_sum / count, moved_sum / count
        cross = cross_sum - moved_mean * fixed_sum - fixed_mean * moved_sum + fixed_mean * moved_mean * count
        fixed_var = fixed2_sum - 2 * fixed_mean * fixed_sum + fixed_mean.square() * count
        moved_var = moved2_sum - 2 * moved_mean * moved_sum + moved_mean.square() * count
        loss = -(cross.square() / (fixed_var * moved_var + self.eps))
        return masked_mean(loss, mask)


class GlobalMutualInformation(torch.nn.Module):
    """Differentiable global MI using Gaussian Parzen windows."""

    def __init__(self, bins=32, sigma=0.05):
        super().__init__()
        self.bins = bins
        self.sigma = sigma

    def forward(self, fixed, moved, mask=None):
        centers = torch.linspace(0, 1, self.bins, device=fixed.device, dtype=fixed.dtype)
        values = []
        for batch_idx in range(fixed.shape[0]):
            x, y = fixed[batch_idx].reshape(-1), moved[batch_idx].reshape(-1)
            if mask is not None:
                valid = mask[batch_idx].reshape(-1) > 0.5
                x, y = x[valid], y[valid]
            # Full 3D volumes contain millions of voxels. Uniformly striding them
            # keeps the Parzen estimate tractable without changing its objective.
            stride = max(1, x.numel() // 65536)
            x, y = x[::stride], y[::stride]
            px = torch.softmax(-0.5 * ((x[:, None] - centers) / self.sigma).square(), dim=1)
            py = torch.softmax(-0.5 * ((y[:, None] - centers) / self.sigma).square(), dim=1)
            joint = px.T @ py
            joint = joint / joint.sum().clamp_min(1e-8)
            product = joint.sum(1, keepdim=True) @ joint.sum(0, keepdim=True)
            values.append(-(joint * torch.log((joint + 1e-8) / (product + 1e-8))).sum())
        return torch.stack(values).mean()


class MSE(torch.nn.Module):
    def forward(self, fixed, moved, mask=None):
        return masked_mean((fixed - moved).square(), mask)


def masked_mean(value, mask):
    if mask is None:
        return value.mean()
    return (value * mask).sum() / mask.sum().clamp_min(1)


def make_image_loss(name, ncc_win=9, mi_bins=32):
    if name == 'mse':
        return MSE()
    if name == 'ncc':
        return LocalNCC(win=ncc_win)
    if name == 'mi':
        return GlobalMutualInformation(bins=mi_bins)
    raise ValueError(f'Unsupported image loss: {name}')


def gradient_loss(flow):
    diffs = [
        flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :],
        flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :],
        flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1],
    ]
    return sum(diff.square().mean() for diff in diffs) / len(diffs)


def foreground_mask(image):
    return (image.abs() > 1e-6).float()


def mean_dice(pred, target):
    labels = np.union1d(np.unique(pred), np.unique(target))
    labels = labels[labels != 0]
    scores = []
    for label in labels:
        pred_label, target_label = pred == label, target == label
        denom = pred_label.sum() + target_label.sum()
        if denom:
            scores.append(2.0 * np.logical_and(pred_label, target_label).sum() / denom)
    return float(np.mean(scores)) if scores else 0.0


def negative_jacobian_fraction(flow):
    disp = flow.detach().cpu().numpy()
    fractions = []
    for sample in disp:
        d0 = np.gradient(sample[0], axis=0)
        d1 = np.gradient(sample[0], axis=1)
        d2 = np.gradient(sample[0], axis=2)
        e0 = np.gradient(sample[1], axis=0)
        e1 = np.gradient(sample[1], axis=1)
        e2 = np.gradient(sample[1], axis=2)
        f0 = np.gradient(sample[2], axis=0)
        f1 = np.gradient(sample[2], axis=1)
        f2 = np.gradient(sample[2], axis=2)
        det = (
            (1 + d0) * ((1 + e1) * (1 + f2) - e2 * f1)
            - d1 * (e0 * (1 + f2) - e2 * f0)
            + d2 * (e0 * f1 - (1 + e1) * f0)
        )
        fractions.append(float(np.mean(det <= 0)))
    return float(np.mean(fractions))


def train_epoch(model, loader, optimizer, scaler, image_loss, lambda_param, device, amp, mask_mode='none'):
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    valid_steps = 0
    for batch in loader:
        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        mask = batch.get('target_mask')
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        elif mask_mode == 'auto':
            mask = foreground_mask(target)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            output = model(source, target, return_pos_flow=True)
        moved = output['moved_vol'].float()
        velocity = output['preint_flow'].float()
        image = image_loss(target.float(), moved, mask)
        smooth = gradient_loss(velocity)
        loss = image + lambda_param * smooth
        if not torch.isfinite(loss):
            print('[WARN] Skipping a non-finite training step.')
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        totals += (loss.item(), image.item(), smooth.item())
        valid_steps += 1
    if not valid_steps:
        raise RuntimeError('No finite training steps completed.')
    return totals / valid_steps


@torch.no_grad()
def validate(model, loader, image_loss, lambda_param, device, mask_mode='none'):
    model.eval()
    totals = np.zeros(5, dtype=np.float64)
    steps = 0
    for batch in loader:
        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        output = model(source, target, return_pos_flow=True)
        moved, velocity, flow = output['moved_vol'].float(), output['preint_flow'].float(), output['pos_flow'].float()
        mask = batch.get('target_mask')
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        elif mask_mode == 'auto':
            mask = foreground_mask(target)
        image = image_loss(target.float(), moved, mask)
        smooth = gradient_loss(velocity)
        dice = 0.0
        if 'source_label' in batch and 'target_label' in batch:
            interpolation_mode = model.spatial_trans.mode
            model.spatial_trans.mode = 'nearest'
            warped = model.spatial_trans(batch['source_label'].float().to(device), flow)
            model.spatial_trans.mode = interpolation_mode
            dice = mean_dice(np.rint(warped.cpu().numpy()), batch['target_label'].numpy())
        totals += (image.item() + lambda_param * smooth.item(), image.item(), smooth.item(), dice, negative_jacobian_fraction(flow))
        steps += 1
    return totals / max(steps, 1)


def run_training(args, model, train_loader, val_loader, image_loss, device):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup)),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp = device == 'cuda' and not args.disable_amp
    if hasattr(torch.amp, 'GradScaler'):
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=amp)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = float('inf')
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as handle:
        csv.writer(handle).writerow(
            ['epoch', 'train_loss', 'train_image', 'train_smooth', 'val_loss', 'val_image', 'val_smooth', 'val_dice', 'val_neg_jac', 'lr']
        )
    for epoch in range(args.start_epoch, args.epochs + 1):
        train = train_epoch(model, train_loader, optimizer, scaler, image_loss, args.lambda_param, device, amp, args.loss_mask_mode)
        val = validate(model, val_loader, image_loss, args.lambda_param, device, args.loss_mask_mode) if val_loader else np.zeros(5)
        current = float(val[0] if val_loader else train[0])
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'inshape': tuple(model.spatial_trans.grid.shape[2:]),
            'integration_steps': 7,
            'args': vars(args),
        }
        if current < best:
            best = current
            torch.save(checkpoint, output)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, output.with_name(f'{output.stem}_epoch{epoch:04d}{output.suffix}'))
        lr = optimizer.param_groups[0]['lr']
        with csv_path.open('a', newline='') as handle:
            csv.writer(handle).writerow([epoch, *train, *val, lr])
        print(
            f'Epoch {epoch:03d}/{args.epochs} | train {train[0]:.6f} '
            f'(img {train[1]:.6f}, smooth {train[2]:.6f}) | '
            f'val {val[0]:.6f}, dice {val[3]:.4f}, neg_jac {val[4]:.6f} | lr {lr:.2e}'
        )
        scheduler.step()
