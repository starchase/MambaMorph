#!/usr/bin/env python3
"""Shared utilities for unsupervised diffeomorphic MambaMorph training."""

import csv
import datetime
import random
import shlex
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
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


def _binary_dilate_3d(mask, kernel_size):
    return F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _binary_erode_3d(mask, kernel_size):
    return 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def foreground_mask(image):
    bg_val = image.amin(dim=(2, 3, 4), keepdim=True)
    mask = (image > (bg_val + 1e-3)).float()
    mask = _binary_dilate_3d(mask, kernel_size=5)
    mask = _binary_erode_3d(mask, kernel_size=5)
    mask = _binary_dilate_3d(mask, kernel_size=3)
    return mask.clamp_(0.0, 1.0)


def dice_scores(pred, target, labels=None):
    if labels is None:
        labels = np.union1d(np.unique(pred), np.unique(target))
    labels = np.asarray(labels)
    labels = labels[labels > 0.5]
    scores = []
    for label in labels:
        pred_label, target_label = pred == label, target == label
        denom = pred_label.sum() + target_label.sum()
        if denom:
            scores.append(2.0 * np.logical_and(pred_label, target_label).sum() / denom)
    return np.asarray(scores, dtype=np.float64)


def mean_dice(pred, target, labels=None):
    scores = dice_scores(pred, target, labels=labels)
    return float(scores.mean()) if scores.size else 0.0


def jacobian_determinant(flow):
    d0 = np.gradient(flow[0], axis=0)
    d1 = np.gradient(flow[0], axis=1)
    d2 = np.gradient(flow[0], axis=2)
    e0 = np.gradient(flow[1], axis=0)
    e1 = np.gradient(flow[1], axis=1)
    e2 = np.gradient(flow[1], axis=2)
    f0 = np.gradient(flow[2], axis=0)
    f1 = np.gradient(flow[2], axis=1)
    f2 = np.gradient(flow[2], axis=2)
    return (
        (1 + d0) * ((1 + e1) * (1 + f2) - e2 * f1)
        - d1 * (e0 * (1 + f2) - e2 * f0)
        + d2 * (e0 * f1 - (1 + e1) * f0)
    )


def negative_jacobian_fraction(flow):
    disp = flow.detach().cpu().numpy()
    fractions = []
    for sample in disp:
        det = jacobian_determinant(sample)
        fractions.append(float(np.mean(det <= 0)))
    return float(np.mean(fractions))


def resolve_loss_mask(batch, flow, target, model, device, mode='manual'):
    if mode == 'none':
        return None
    if mode == 'auto':
        return foreground_mask(target.float())
    if mode != 'manual':
        raise ValueError(f'Unsupported loss mask mode: {mode}')

    mask_parts = []
    target_mask = batch.get('target_mask')
    if target_mask is not None:
        mask_parts.append((target_mask.to(device, non_blocking=True) > 0.5).float())

    source_mask = batch.get('source_mask')
    if source_mask is not None:
        old_mode = model.spatial_trans.mode
        model.spatial_trans.mode = 'nearest'
        with torch.no_grad():
            warped = model.spatial_trans(source_mask.float().to(device, non_blocking=True), flow.detach())
        model.spatial_trans.mode = old_mode
        mask_parts.append((warped > 0.5).float())

    if not mask_parts:
        return None
    return torch.clamp(sum(mask_parts), min=0.0, max=1.0)


def create_grad_scaler(device, enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler(device, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled) if device == 'cuda' else None


def train_epoch(model, loader, optimizer, scaler, image_loss, lambda_param, device, amp, mask_mode='none'):
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    valid_steps = 0
    grad_norm_total = 0.0
    grad_norm_steps = 0
    effective_update_steps = 0
    last_image = 0.0
    last_smooth = 0.0
    ref_param = next((p for p in model.parameters() if p.requires_grad), None)
    for batch in loader:
        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            output = model(source, target, return_pos_flow=True)
        moved = output['moved_vol'].float()
        velocity = output['preint_flow'].float()
        flow = output['pos_flow'].float()
        mask = resolve_loss_mask(batch, flow, target, model, device, mask_mode)
        image = image_loss(target.float(), moved, mask)
        smooth = gradient_loss(velocity)
        loss = image + lambda_param * smooth
        if not torch.isfinite(loss):
            print('[WARN] Skipping a non-finite training step.')
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_sq_sum = 0.0
        has_nonfinite_grad = False
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                if not torch.isfinite(grad).all():
                    has_nonfinite_grad = True
                    break
                grad_sq_sum += torch.sum(grad * grad).item()
        if has_nonfinite_grad:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        if grad_sq_sum > 0 and np.isfinite(grad_sq_sum):
            grad_norm_total += grad_sq_sum ** 0.5
            grad_norm_steps += 1
        before_ref = ref_param.detach().clone() if ref_param is not None else None
        scaler.step(optimizer)
        scaler.update()
        if before_ref is not None and (ref_param.detach() - before_ref).abs().mean().item() > 0:
            effective_update_steps += 1
        totals += (loss.item(), image.item(), smooth.item())
        last_image, last_smooth = image.item(), smooth.item()
        valid_steps += 1
    if not valid_steps:
        raise RuntimeError('No finite training steps completed.')
    avg = totals / valid_steps
    avg_grad_norm = grad_norm_total / grad_norm_steps if grad_norm_steps else 0.0
    update_ratio = effective_update_steps / max(valid_steps, 1)
    return avg[0], 0.0, last_image, last_smooth, avg_grad_norm, update_ratio


@torch.no_grad()
def warp_labels(model, labels, flow):
    old_mode = model.spatial_trans.mode
    model.spatial_trans.mode = 'nearest'
    warped = model.spatial_trans(labels.float(), flow)
    model.spatial_trans.mode = old_mode
    return warped


def compute_hd95(ground_truth, prediction, spacing=None):
    if ground_truth.sum() == 0 or prediction.sum() == 0:
        return np.nan
    import scipy.ndimage
    from scipy.spatial import cKDTree

    pred_border = prediction ^ scipy.ndimage.binary_erosion(prediction)
    gt_border = ground_truth ^ scipy.ndimage.binary_erosion(ground_truth)
    pts_pred = np.argwhere(pred_border)
    pts_gt = np.argwhere(gt_border)
    if pts_pred.shape[0] == 0 or pts_gt.shape[0] == 0:
        return np.nan
    if spacing is not None:
        pts_pred = pts_pred * np.asarray(spacing)
        pts_gt = pts_gt * np.asarray(spacing)
    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)
    dist_pred_to_gt, _ = tree_gt.query(pts_pred, k=1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt, k=1)
    return max(np.percentile(dist_pred_to_gt, 95), np.percentile(dist_gt_to_pred, 95))


@torch.no_grad()
def validate(model, loader, image_loss, lambda_param, device, mask_mode='none'):
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    steps = 0
    for batch in loader:
        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        start = time.time()
        if device == 'cuda':
            torch.cuda.synchronize()
        output = model(source, target, return_pos_flow=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        total_time += (time.time() - start) / source.shape[0]
        moved, velocity, flow = output['moved_vol'].float(), output['preint_flow'].float(), output['pos_flow'].float()
        mask = resolve_loss_mask(batch, flow, target, model, device, mask_mode)
        image = image_loss(target.float(), moved, mask)
        smooth = gradient_loss(velocity)
        total_loss += image.item() + lambda_param * smooth.item()
        steps += 1
    return total_loss / max(steps, 1), 0.0, 0.0, 0.0, total_time / max(steps, 1), 0.0, 0.0


@torch.no_grad()
def test_evaluate(model, loader, image_loss, lambda_param, device, fast=False, mask_mode='manual'):
    model.eval()
    total_loss = 0.0
    total_time = 0.0
    total_reg_time = 0.0
    total_mag = 0.0
    total_neg_jac = 0.0
    num_jac = 0
    dice_values = []
    hd95_values = []
    raw_results = []
    label_scores = {}
    best_dice = -1.0
    best_idx = None
    num_batches = 0

    for batch in loader:
        source = batch['source'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)

        if device == 'cuda':
            torch.cuda.synchronize()
        start_reg = time.time()
        _ = model(source, target, return_pos_flow=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        total_reg_time += (time.time() - start_reg) / source.shape[0]

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        output = model(source, target, return_pos_flow=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        total_time += (time.time() - start) / source.shape[0]

        moved = output['moved_vol'].float()
        velocity = output['preint_flow'].float()
        flow = output['pos_flow'].float()
        mask = resolve_loss_mask(batch, flow, target, model, device, mask_mode)
        image = image_loss(target.float(), moved, mask)
        smooth = gradient_loss(velocity)
        total_loss += (image + lambda_param * smooth).item()
        total_mag += torch.sqrt(torch.sum(flow * flow, dim=1)).mean().item()
        num_batches += 1

        flow_np = flow.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        batch_jacs = []
        if not fast:
            for i in range(flow_np.shape[0]):
                jac_det = jacobian_determinant(flow_np[i])
                valid_mask = target_np[i, 0] > 0.01
                valid = valid_mask.sum()
                if valid > 0:
                    batch_jacs.append(float(np.sum((jac_det <= 0) & valid_mask) / valid))
            if batch_jacs:
                total_neg_jac += float(np.mean(batch_jacs))
                num_jac += 1

        if 'source_label' in batch and 'target_label' in batch:
            source_label = batch['source_label'].to(device, non_blocking=True)
            target_label = batch['target_label'].to(device, non_blocking=True)
            warped_label = warp_labels(model, source_label, flow)
            wl_np = np.rint(warped_label.cpu().numpy())
            tl_np = target_label.cpu().numpy()
            sl_np = source_label.cpu().numpy()
            filenames = batch.get('filename', [''] * source.shape[0])
            if isinstance(filenames, str):
                filenames = [filenames]
            for b in range(wl_np.shape[0]):
                labels = np.intersect1d(np.unique(sl_np[b]), np.unique(tl_np[b]))
                labels = labels[labels > 0.5]
                scores = dice_scores(wl_np[b], tl_np[b], labels=labels)
                sample_dice = float(scores.mean()) if scores.size else 0.0
                label_dice = {}
                for idx, label in enumerate(labels):
                    value = float(scores[idx]) if idx < len(scores) else 0.0
                    key = int(label)
                    label_dice[key] = value
                    label_scores.setdefault(key, []).append(value)
                sample_hd95 = np.nan
                if not fast and labels.size:
                    hds = []
                    for label in labels:
                        hd = compute_hd95(tl_np[b, 0] == label, wl_np[b, 0] == label)
                        if not np.isnan(hd):
                            hds.append(hd)
                    if hds:
                        sample_hd95 = float(np.mean(hds))
                        hd95_values.append(sample_hd95)
                sample_jac = batch_jacs[b] if b < len(batch_jacs) else np.nan
                sample_idx = len(raw_results)
                raw_results.append({
                    'sample_idx': sample_idx,
                    'filename': filenames[b] if b < len(filenames) else '',
                    'dice': sample_dice,
                    'hd95': sample_hd95,
                    'jac': sample_jac,
                    'label_dice': label_dice,
                })
                dice_values.append(sample_dice)
                if sample_dice > best_dice:
                    best_dice = sample_dice
                    best_idx = sample_idx

    avg_dice = float(np.mean(dice_values)) if dice_values else 0.0
    std_dice = float(np.std(dice_values)) if dice_values else 0.0
    avg_hd95 = float(np.mean(hd95_values)) if hd95_values else 0.0
    std_hd95 = float(np.std(hd95_values)) if hd95_values else 0.0
    jac_values = [r['jac'] for r in raw_results if 'jac' in r and not np.isnan(r['jac'])]
    avg_neg_jac = total_neg_jac / num_jac if num_jac else 0.0
    std_neg_jac = float(np.std(jac_values)) if jac_values else 0.0
    avg_label = {k: float(np.mean(v)) for k, v in label_scores.items()}
    std_label = {k: float(np.std(v)) for k, v in label_scores.items()}
    return (
        total_loss / max(num_batches, 1),
        0.0,
        avg_dice,
        std_dice,
        avg_hd95,
        std_hd95,
        total_time / max(num_batches, 1),
        total_reg_time / max(num_batches, 1),
        avg_neg_jac,
        std_neg_jac,
        total_mag / max(num_batches, 1),
        best_idx,
        avg_label,
        std_label,
        raw_results,
    )


def check_early_stopping(loss_history, patience=20, warm_start_steps=10):
    if len(loss_history) < warm_start_steps:
        return False
    best_idx = int(np.argmin(loss_history))
    return len(loss_history) - 1 - best_idx >= patience


def _slice(tensor, z_idx):
    if tensor is None:
        return None
    return np.rot90(tensor[:, :, :, :, z_idx].detach().cpu().numpy()[0, 0])


def boundary_magnitude(volume):
    dz = torch.zeros_like(volume)
    dy = torch.zeros_like(volume)
    dx = torch.zeros_like(volume)
    dz[:, :, 1:, :, :] = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
    dy[:, :, :, 1:, :] = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
    dx[:, :, :, :, 1:] = volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]
    mag = torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-8)
    max_val = mag.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return mag / max_val


@torch.no_grad()
def save_qualitative_results(model, dataset, output_dir, epoch, device='cuda', suffix='', best_sample_idx=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    if 'test' in suffix:
        for idx in range(len(dataset)):
            items.append((dataset[idx].get('filename', f'{idx:04d}'), dataset[idx]))
    elif len(dataset):
        items.append(('default', dataset[0]))
        if best_sample_idx is not None and best_sample_idx < len(dataset):
            items.append(('best_dice', dataset[best_sample_idx]))

    for tag, sample in items:
        source = sample['source'].unsqueeze(0).to(device)
        target = sample['target'].unsqueeze(0).to(device)
        source_label = sample.get('source_label')
        target_label = sample.get('target_label')
        source_label = source_label.unsqueeze(0).to(device) if source_label is not None else None
        target_label = target_label.unsqueeze(0).to(device) if target_label is not None else None
        output = model(source, target, return_pos_flow=True)
        moved = output['moved_vol'].float()
        flow = output['pos_flow'].float()
        warped_label = warp_labels(model, source_label, flow) if source_label is not None else None
        source_boundary = boundary_magnitude(source.float())
        target_boundary = boundary_magnitude(target.float())
        moved_boundary = boundary_magnitude(moved.float())
        active_mask = foreground_mask(target.float())

        slice_idx = source.shape[4] // 2
        if source_label is not None and target_label is not None:
            best_score = -1.0
            best_idx = slice_idx
            search_indices = sorted(range(target_label.shape[4]), key=lambda idx: abs(idx - slice_idx))
            for z_idx in search_indices:
                src_slice_gpu = source_label[0, 0, :, :, z_idx]
                tgt_slice_gpu = target_label[0, 0, :, :, z_idx]
                if src_slice_gpu.sum() == 0 or tgt_slice_gpu.sum() == 0:
                    continue
                src_labels = torch.unique(src_slice_gpu)
                tgt_labels = torch.unique(tgt_slice_gpu)
                src_labels = src_labels[src_labels > 0.5]
                tgt_labels = tgt_labels[tgt_labels > 0.5]
                if len(src_labels) == 0 or len(tgt_labels) == 0:
                    continue
                common_count = sum(1 for label in src_labels if (tgt_labels == label).any())
                avg_slice_dice = 0.0
                if warped_label is not None:
                    warped_slice_gpu = warped_label[0, 0, :, :, z_idx]
                    dice_sum = 0.0
                    dice_n = 0
                    for label in tgt_labels:
                        pred_mask = warped_slice_gpu == label
                        tgt_mask = tgt_slice_gpu == label
                        denom = pred_mask.sum().float() + tgt_mask.sum().float()
                        if denom > 0:
                            dice_sum += (2.0 * (pred_mask & tgt_mask).sum().float() / denom).item()
                            dice_n += 1
                    if dice_n:
                        avg_slice_dice = dice_sum / dice_n
                score = (common_count * 10.0) + len(src_labels) + len(tgt_labels) + (avg_slice_dice * 0.5)
                if score > best_score:
                    best_score = score
                    best_idx = z_idx
            slice_idx = best_idx

        src = _slice(source, slice_idx)
        tgt = _slice(target, slice_idx)
        mov = _slice(moved, slice_idx)
        src_boundary = _slice(source_boundary, slice_idx)
        tgt_boundary = _slice(target_boundary, slice_idx)
        mov_boundary = _slice(moved_boundary, slice_idx)
        mask_slice = _slice(active_mask, slice_idx)
        src_lbl = _slice(source_label, slice_idx)
        tgt_lbl = _slice(target_label, slice_idx)
        wrp_lbl = _slice(warped_label, slice_idx)
        disp = flow.detach().cpu().numpy()[0]
        disp_slice = np.rot90(disp[:, :, :, slice_idx], axes=(1, 2))
        jac = np.rot90(jacobian_determinant(disp)[:, :, slice_idx])
        has_labels = src_lbl is not None and tgt_lbl is not None

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        for ax in axes.ravel():
            ax.axis('off')

        axes[0, 0].imshow(src - tgt, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 0].set_title('Diff: Source - Target')
        axes[0, 1].imshow(mov - tgt, cmap='bwr', vmin=-1, vmax=1)
        axes[0, 1].set_title('Diff: Deformed - Target')

        h, w = src.shape
        grid_spacing = 10
        axes[0, 2].imshow(np.zeros_like(src), cmap='gray', vmin=0, vmax=1)
        for x_idx in range(0, w, grid_spacing):
            if x_idx < disp_slice.shape[2]:
                x_plot = x_idx + disp_slice[2, :, x_idx]
                y_plot = np.arange(h) + disp_slice[1, :, x_idx]
                axes[0, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
        for y_idx in range(0, h, grid_spacing):
            if y_idx < disp_slice.shape[1]:
                x_plot = np.arange(w) + disp_slice[2, y_idx, :]
                y_plot = y_idx + disp_slice[1, y_idx, :]
                axes[0, 2].plot(x_plot, y_plot, 'w-', linewidth=0.8, alpha=0.9)
        axes[0, 2].set_title('Deformed Grid')
        axes[0, 2].set_ylim(h, 0)
        axes[0, 2].set_xlim(0, w)

        jac_vis = np.zeros((*jac.shape, 3), dtype=np.float32)
        jac_vis[jac < 0] = (1.0, 0.0, 0.0)
        jac_vis[(jac >= 0) & (jac <= 1)] = (0.4, 0.8, 0.4)
        jac_vis[jac > 1] = (0.4, 0.6, 0.9)
        axes[0, 3].imshow(jac_vis)
        axes[0, 3].set_title('Jacobian Determinant')

        dx, dy, dz = disp_slice[2], disp_slice[1], disp_slice[0]
        max_mag = float(np.max(np.abs(disp_slice)) + 1e-5)
        flow_vis = np.zeros((h, w, 3), dtype=np.float32)
        flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5
        flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5
        flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5
        axes[1, 0].imshow(np.clip(flow_vis, 0, 1))
        axes[1, 0].set_title('RGB Displacement')

        from matplotlib.colors import hsv_to_rgb

        legend_ax = axes[1, 1]
        legend_ax.clear()
        legend_ax.axis('off')
        legend_ax.set_aspect('equal')
        legend_ax.set_xlim(-1.2, 1.2)
        legend_ax.set_ylim(-1.2, 1.2)
        x_wheel = np.linspace(-0.12, 0.12, 100)
        y_wheel = np.linspace(-0.12, 0.12, 100)
        xw, yw = np.meshgrid(x_wheel, y_wheel)
        rw = np.sqrt(xw ** 2 + yw ** 2)
        tw = np.arctan2(yw, xw)
        tw[tw < 0] += 2 * np.pi
        rgb_wheel = hsv_to_rgb(np.stack((tw / (2 * np.pi), np.ones_like(tw), np.ones_like(tw)), axis=-1))
        rgba_wheel = np.concatenate([rgb_wheel, (rw <= 0.12)[..., None].astype(float)], axis=-1)
        legend_ax.imshow(rgba_wheel, extent=[-0.12, 0.12, -0.12, 0.12], origin='lower')
        vec_x = np.array([0.5, -0.2])
        vec_y = np.array([-0.4, -0.25])
        vec_z = np.array([0.0, 0.5])
        scale = 0.312
        for vec, label, valign in [(vec_x, 'x', 'center'), (vec_y, 'y', 'center'), (vec_z, 'z', 'bottom')]:
            legend_ax.arrow(0, 0, vec[0] * scale, vec[1] * scale, head_width=0.024, head_length=0.03, fc='black', ec='black')
            legend_ax.text(vec[0] * scale * 1.6, vec[1] * scale * (1.4 if label == 'z' else 1.6), label, fontweight='bold', fontsize=16, ha='center', va=valign)
        legend_ax.text(0, -0.35, f'[{ -max_mag:.2f}, {max_mag:.2f}]', ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        base_colors = {
            1: '#8B0000',
            2: '#228B22',
            3: '#4682B4',
            4: '#DAA520',
            5: '#008B8B',
        }
        bright_colors = {
            1: '#FF0000',
            2: '#00FF00',
            3: '#0000FF',
            4: '#FFD700',
            5: '#00FFFF',
        }
        text_colors = {
            1: '#A52A2A',
            2: '#2E8B57',
            3: '#4682B4',
            4: '#CD853F',
            5: '#20B2AA',
        }

        def draw_dice_text(ax, dice_results):
            if not dice_results:
                return
            dice_results.sort(key=lambda item: item[0], reverse=True)
            spacing = 0.20
            total_width = len(dice_results) * spacing
            x_offset = (1.0 - total_width) / 2.0 + spacing / 2.0
            for dice, color in dice_results:
                ax.text(x_offset, 0.02, f'{dice:.2f}', color=color, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='center')
                x_offset += spacing

        axes[1, 2].imshow(mov, cmap='gray')
        if has_labels and wrp_lbl is not None:
            labels = np.unique(np.concatenate([tgt_lbl, wrp_lbl]))
            labels = labels[labels > 0]
            dice_results = []
            for label in labels:
                target_mask = tgt_lbl == label
                warped_mask = wrp_lbl == label
                if np.any(target_mask):
                    axes[1, 2].contour(target_mask, colors=[base_colors.get(int(label), 'white')], linewidths=1.5, linestyles='solid')
                if np.any(warped_mask):
                    axes[1, 2].contour(warped_mask, colors=[bright_colors.get(int(label), 'white')], linewidths=1.5, linestyles='solid')
                denom = warped_mask.sum() + target_mask.sum()
                if denom > 0:
                    dice = 2.0 * np.logical_and(warped_mask, target_mask).sum() / denom
                    dice_results.append((dice, text_colors.get(int(label), 'white')))
            draw_dice_text(axes[1, 2], dice_results)
        axes[1, 2].set_title('Result vs GT')
        axes[1, 3].axis('off')

        def plot_label_contour(ax, bg_img, label_img, title, color_lookup, target_img=None):
            ax.imshow(bg_img, cmap='gray')
            if label_img is not None:
                labels = np.unique(np.concatenate([label_img, target_img])) if target_img is not None else np.unique(label_img)
                labels = labels[labels > 0]
                dice_results = []
                for label in labels:
                    mask = label_img == label
                    if np.any(mask):
                        ax.contour(mask, colors=[color_lookup.get(int(label), 'white')], linewidths=1.2)
                    if target_img is not None:
                        target_mask = target_img == label
                        denom = mask.sum() + target_mask.sum()
                        if denom > 0:
                            dice = 2.0 * np.logical_and(mask, target_mask).sum() / denom
                            dice_results.append((dice, text_colors.get(int(label), 'white')))
                draw_dice_text(ax, dice_results)
            ax.set_title(title)
            ax.axis('off')

        if has_labels:
            plot_label_contour(axes[2, 0], src, src_lbl, 'Source + Labels', bright_colors, target_img=tgt_lbl)
            plot_label_contour(axes[2, 1], tgt, tgt_lbl, 'Target + Labels', base_colors)
            plot_label_contour(axes[2, 2], mov, wrp_lbl, 'Deformed + Labels', bright_colors, target_img=tgt_lbl)
        axes[2, 3].axis('off')

        boundary_vmax = max(float(np.max(src_boundary)), float(np.max(tgt_boundary)), float(np.max(mov_boundary)), 1e-6)
        axes[3, 0].imshow(src_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 0].set_title('Source Boundary Map')
        axes[3, 1].imshow(tgt_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 1].set_title('Target Boundary Map')
        axes[3, 2].imshow(mov_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
        axes[3, 2].set_title('Warped Boundary Map')
        axes[3, 3].imshow(tgt, cmap='gray')
        axes[3, 3].imshow(mask_slice, cmap='autumn', alpha=0.45, vmin=0.0, vmax=1.0)
        axes[3, 3].set_title('Foreground Mask')

        plt.suptitle(f'Epoch {epoch} - Sample {tag} (Slice Z={slice_idx})', fontsize=16)
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except UserWarning:
            pass
        safe_tag = str(tag).replace('/', '_').replace('\\', '_')
        plt.savefig(output_dir / f'vis_epoch_{epoch:04d}{suffix}_{safe_tag}.png')
        plt.close(fig)


def plot_history(log_file, output_dir):
    epochs, train_loss, val_loss, test_loss, test_dice, test_jac, test_mag = [], [], [], [], [], [], []
    with open(log_file, 'r', newline='') as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row.get('val_loss', 0.0)))
            test_loss.append(float(row.get('test_loss', 0.0)))
            test_dice.append(float(row.get('test_dice', 0.0)))
            test_jac.append(float(row.get('test_jac', 0.0)))
            test_mag.append(float(row.get('test_mag', 0.0)))
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    if any(value != 0 for value in test_loss):
        plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, test_dice, label='Test Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, test_jac, label='Test Neg Jac')
    plt.title('Negative Jacobian Ratio (Folding)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_mag, label='Test Mag')
    plt.title('Deformation Magnitude')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png')
    plt.close()


def run_training(args, model, train_loader, val_loader, test_loader, image_loss, device):
    utc_now = datetime.datetime.utcnow()
    beijing_time = utc_now + datetime.timedelta(hours=8)
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    input_output = Path(args.output)
    loss_tag = args.loss.lower()
    run_dir = input_output.parent / f'{input_output.stem}_{loss_tag}_{timestamp}'
    output = run_dir / f'{input_output.stem}_{loss_tag}{input_output.suffix}'
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Output directory for this run: {output.parent}')

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
    scaler = create_grad_scaler(device, enabled=amp)
    best = float('inf')
    best_test_dice = 0.0
    loss_history = []
    epoch_times = []
    csv_path = output.parent / 'train_log.csv'
    config_path = output.parent / 'config.txt'

    with config_path.open('w') as handle:
        handle.write('Training Configuration:\n')
        handle.write(f'Timestamp: {timestamp}\n')
        handle.write(f'Start Time: {timestamp}\n')
        handle.write(f'Dataset: {Path(args.ct_dir).parent.parent.parent.name}\n')
        handle.write(f'Device: {device}\n')
        handle.write(f'Epochs: {args.epochs}\n')
        handle.write(f'Start Epoch: {args.start_epoch}\n')
        handle.write(f'Resume: {args.resume}\n')
        handle.write(f'Seed: {args.seed}\n')
        handle.write(f'Batch Size: {args.batch_size}\n')
        handle.write(f'Image Loss: {args.loss}\n')
        handle.write(f'NCC Window: {args.ncc_win}\n')
        handle.write(f'MI Bins: {args.mi_bins}\n')
        handle.write(f'Loss Mask Mode: {args.loss_mask_mode}\n')
        handle.write(f'Lambda: {args.lambda_param}\n')
        handle.write(f'LR: {args.lr}\n')
        handle.write('Integration Steps: 7\n')
        handle.write(f'Unpaired: {getattr(args, "unpaired", False)}\n')
        handle.write(f'Val Paired: {getattr(args, "val_paired", False)}\n')
        handle.write('Model Architecture: MambaMorph\n')
        handle.write(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n')
        handle.write(f'Output Path: {output}\n')
        handle.write(f'Command: {" ".join(shlex.quote(arg) for arg in sys.argv)}\n')
        handle.write(f'Remark: {getattr(args, "remark", "")}\n')
        handle.write(f'Arguments: {vars(args)}\n')

    with csv_path.open('w', newline='') as handle:
        csv.writer(handle).writerow(
            [
                'epoch', 'train_loss', 'test_dice', 'test_hd95', 'test_jac', 'test_mag',
                'train_grad_norm', 'train_update_ratio', 'val_loss', 'test_loss',
                'test_dice_std', 'test_hd95_std', 'test_jac_std', 'test_time_sec',
                'test_reg_time_sec', 'test_dice_per_label', 'test_dice_per_label_std',
            ]
        )
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    print(f'Training for {args.epochs} epochs...')
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start = time.time()
        avg_loss, train_boundary_loss, last_img_loss, last_grad_loss, avg_grad_norm, update_ratio = train_epoch(
            model, train_loader, optimizer, scaler, image_loss, args.lambda_param, device, amp, args.loss_mask_mode
        )
        val_loss, val_boundary_loss, val_dice, val_hd95, val_time, val_jac, val_mag = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        current = avg_loss
        if val_loader:
            val_loss, val_boundary_loss, val_dice, val_hd95, val_time, val_jac, val_mag = validate(
                model, val_loader, image_loss, args.lambda_param, device, args.loss_mask_mode
            )
            current = val_loss
            print(
                f'Epoch {epoch} | TrainTotal: {avg_loss:.4f} '
                f'(Img: {last_img_loss:.4f}, Grad: {last_grad_loss:.6f}) | '
                f'GradNorm: {avg_grad_norm:.6f}, UpdateRatio: {update_ratio:.2%}'
            )
            print(f'         | Val Loss: {val_loss:.4f} (Fast Val, Loss Only)')
        else:
            print(f'Epoch {epoch}, Train Loss: {avg_loss:.6f}, GradNorm: {avg_grad_norm:.6f}, UpdateRatio: {update_ratio:.2%}')

        test_loss = test_dice = test_dice_std = test_hd95 = test_hd95_std = 0.0
        test_jac = test_jac_std = test_mag = test_time = test_reg_time = 0.0
        test_best_idx = None
        test_label_avg = {}
        test_label_std = {}
        is_best_test_dice = False
        run_full_metrics = epoch in [1, 3, 5, 7] or epoch % 10 == 0
        if test_loader:
            (
                test_loss, _, test_dice, test_dice_std, test_hd95, test_hd95_std,
                test_time, test_reg_time, test_jac, test_jac_std, test_mag,
                test_best_idx, test_label_avg, test_label_std, _,
            ) = test_evaluate(model, test_loader, image_loss, args.lambda_param, device, fast=not run_full_metrics, mask_mode=args.loss_mask_mode)
            if test_dice > best_test_dice:
                best_test_dice = test_dice
                is_best_test_dice = True
                print(f'  [Monitor] * New best Test Dice: {best_test_dice:.6f} *')
                best_path = output.parent / f'{output.stem}_best_test.pt'
                torch.save(model.state_dict(), best_path)
                print(f'  [Monitor] Saved new best model to {best_path.name}')
            suffix = '' if run_full_metrics else ' (Fast Test)'
            print(
                f'  [Monitor] Test Dice: {test_dice:.6f}+/-{test_dice_std:.6f}, '
                f'HD95: {test_hd95:.6f}+/-{test_hd95_std:.6f}, Loss: {test_loss:.6f}, '
                f'Jac: {test_jac:.6f}+/-{test_jac_std:.6f}, Time: {test_time:.4f}s{suffix}'
            )

        label_avg_str = '{' + '; '.join(f'{k}: {v:.5f}' for k, v in test_label_avg.items()) + '}' if test_label_avg else ''
        label_std_str = '{' + '; '.join(f'{k}: {v:.5f}' for k, v in test_label_std.items()) + '}' if test_label_std else ''
        with csv_path.open('a', newline='') as handle:
            csv.writer(handle).writerow([
                epoch,
                f'{avg_loss:.5f}',
                f'{test_dice:.5f}',
                f'{test_hd95:.5f}',
                f'{test_jac:.5f}',
                f'{test_mag:.5f}',
                f'{avg_grad_norm:.5f}',
                f'{update_ratio:.5f}',
                f'{val_loss:.5f}',
                f'{test_loss:.5f}',
                f'{test_dice_std:.5f}',
                f'{test_hd95_std:.5f}',
                f'{test_jac_std:.5f}',
                f'{test_time:.5f}',
                f'{test_reg_time:.5f}',
                label_avg_str,
                label_std_str,
            ])

        loss_history.append(current)
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
            print(f'New best val loss: {best:.6f}.')

        do_visualization = is_best_test_dice or run_full_metrics or (getattr(args, 'visualize_every', 0) and epoch % args.visualize_every == 0)
        if do_visualization:
            if test_loader and hasattr(test_loader.dataset, '__getitem__'):
                save_qualitative_results(model, test_loader.dataset, output.parent, epoch, device=device, suffix='_test', best_sample_idx=test_best_idx)
            elif val_loader and hasattr(val_loader.dataset, '__getitem__'):
                save_qualitative_results(model, val_loader.dataset, output.parent, epoch, device=device, suffix='_val')

        if check_early_stopping(loss_history, patience=args.patience, warm_start_steps=args.warm_start):
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % args.save_every == 0:
            ckpt = output.parent / f'{output.stem}_epoch{epoch}.pt'
            torch.save(model.state_dict(), ckpt)
            print(f'Checkpoint saved to {ckpt}')

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0.0
        print(f'Current LR: {lr:.6f} | Epoch Time: {epoch_time:.2f}s | Peak GPU Mem: {peak_mem:.2f} MB')

    final_path = output.parent / f'{output.stem}_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')
    finish_utc = datetime.datetime.utcnow()
    finish_beijing = finish_utc + datetime.timedelta(hours=8)
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    final_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0.0
    with config_path.open('a') as handle:
        handle.write(f'End Time: {finish_beijing.strftime("%Y%m%d_%H%M%S")}\n')
        handle.write(f'Average Epoch Time: {avg_epoch_time:.2f} s\n')
        handle.write(f'Peak GPU Memory: {final_peak_mem:.2f} MB\n')
    try:
        plot_history(csv_path, output.parent)
        print(f'Training curves saved to {output.parent / "training_curves.png"}')
    except Exception as exc:
        print(f'Failed to plot training history: {exc}')
