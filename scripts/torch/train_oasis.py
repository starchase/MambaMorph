#!/usr/bin/env python3
"""Unsupervised diffeomorphic MambaMorph training on OASIS."""

import argparse
import csv
import datetime
import glob
import os
import pickle
import random
import shlex
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mambamorph_train_common import (
    build_model,
    boundary_magnitude,
    foreground_mask,
    gradient_loss,
    jacobian_determinant,
    make_image_loss,
    seed_everything,
)


VOI_LABELS = list(range(1, 36))
VIS_LABELS = [3, 7, 11, 14, 22, 26, 30]


def pkload(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def volume(array, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).unsqueeze(0)


class OASISTrainDataset(Dataset):
    """Match the reference OASISBrainDataset: random moving/fixed train pairs."""

    def __init__(self, paths):
        self.paths = sorted(paths)
        if len(self.paths) < 2:
            raise ValueError('OASIS training requires at least two .pkl files.')
        print(f'Training set loaded: {len(self.paths)} volumes.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        source_path = self.paths[index]
        target_paths = self.paths.copy()
        target_paths.remove(source_path)
        random.shuffle(target_paths)
        source, source_label = pkload(source_path)
        target, target_label = pkload(target_paths[0])
        return (
            volume(source),
            volume(target),
            volume(source_label, torch.long),
            volume(target_label, torch.long),
        )


class OASISValidationDataset(Dataset):
    """Match the reference OASISBrainInferDataset: fixed validation pairs."""

    def __init__(self, paths):
        self.paths = sorted(paths)
        if not self.paths:
            raise ValueError('OASIS validation directory must contain .pkl files.')
        print(f'Validation set loaded: {len(self.paths)} pairs.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        source, target, source_label, target_label = pkload(self.paths[index])
        return (
            volume(source),
            volume(target),
            volume(source_label, torch.long),
            volume(target_label, torch.long),
        )


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def dice_val_voi(pred, target):
    pred_np = pred.detach().cpu().numpy()[0, 0]
    target_np = target.detach().cpu().numpy()[0, 0]
    scores = []
    for label in VOI_LABELS:
        pred_mask = pred_np == label
        target_mask = target_np == label
        denom = pred_mask.sum() + target_mask.sum()
        scores.append((2.0 * np.logical_and(pred_mask, target_mask).sum()) / (denom + 1e-5))
    return float(np.mean(scores))


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
    return float(max(np.percentile(dist_pred_to_gt, 95), np.percentile(dist_gt_to_pred, 95)))


def warp_labels(model, labels, flow):
    old_mode = model.spatial_trans.mode
    model.spatial_trans.mode = 'nearest'
    warped = model.spatial_trans(labels.float(), flow)
    model.spatial_trans.mode = old_mode
    return warped


def image_loss_value(image_loss, loss_name, target, moved, mask=None):
    if loss_name == 'mse':
        value = (target - moved).square()
        if mask is not None:
            return (value * mask).sum() / mask.sum().clamp_min(1)
        return value.mean()
    return image_loss(target, moved, mask)


def boundary_ring_mask(mask, inner_kernel=3, outer_kernel=7):
    inner = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=inner_kernel, stride=1, padding=inner_kernel // 2)
    outer = F.max_pool3d(mask, kernel_size=outer_kernel, stride=1, padding=outer_kernel // 2)
    return (outer - inner).clamp_(0.0, 1.0)


def train_epoch(model, loader, optimizer, scaler, image_loss, loss_name, lambda_param, device, amp, use_mask=False, accumulation_steps=1):
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    valid_batches = 0
    pending_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for source, target, _, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=amp):
            output = model(source, target, return_pos_flow=True)
        moved = output['moved_vol'].float()
        velocity = output['preint_flow'].float()
        target_float = target.float()
        mask = foreground_mask(target_float) if use_mask else None
        img_loss = image_loss_value(image_loss, loss_name, target_float, moved, mask)
        grad_loss = gradient_loss(velocity)
        loss = img_loss + lambda_param * grad_loss
        if not torch.isfinite(loss):
            print(f'[WARN] Skipping non-finite OASIS train batch: img={img_loss.item()} grad={grad_loss.item()} loss={loss.item()}')
            optimizer.zero_grad(set_to_none=True)
            pending_steps = 0
            continue
        scaler.scale(loss / accumulation_steps).backward()
        totals += (loss.item(), img_loss.item(), grad_loss.item())
        valid_batches += 1
        pending_steps += 1
        if pending_steps >= accumulation_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_steps = 0
    if pending_steps:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if valid_batches == 0:
        raise RuntimeError('No finite OASIS training batches completed.')
    return totals / valid_batches


@torch.no_grad()
def validate(model, loader, image_loss, loss_name, lambda_param, device, compute_extra=False, use_mask=False):
    model.eval()
    eval_loss = AverageMeter()
    eval_dsc = AverageMeter()
    eval_hd95 = AverageMeter()
    eval_jac = AverageMeter()
    eval_mag = AverageMeter()
    for source, target, source_label, target_label in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        source_label = source_label.to(device, non_blocking=True)
        target_label = target_label.to(device, non_blocking=True)
        output = model(source, target, return_pos_flow=True)
        moved = output['moved_vol'].float()
        velocity = output['preint_flow'].float()
        flow = output['pos_flow'].float()
        mask = foreground_mask(target.float()) if use_mask else None
        img_loss = image_loss_value(image_loss, loss_name, target.float(), moved, mask)
        grad_loss = gradient_loss(velocity)
        eval_loss.update((img_loss + lambda_param * grad_loss).item(), source.shape[0])
        warped_label = warp_labels(model, source_label, flow)
        dsc = dice_val_voi(torch.round(warped_label).long(), target_label.long())
        eval_dsc.update(dsc, source.shape[0])
        if compute_extra:
            eval_mag.update(torch.sqrt(torch.sum(flow * flow, dim=1)).mean().item(), source.shape[0])
            flow_np = flow.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            batch_jac = 0.0
            for sample_idx in range(flow_np.shape[0]):
                jac_det = jacobian_determinant(flow_np[sample_idx])
                valid = target_np[sample_idx, 0] > 0.01
                valid_sum = valid.sum()
                if valid_sum > 0:
                    batch_jac += float(np.sum((jac_det <= 0) & valid) / valid_sum)
            eval_jac.update(batch_jac / flow_np.shape[0], source.shape[0])
            warped_np = np.rint(warped_label.detach().cpu().numpy())
            target_label_np = target_label.detach().cpu().numpy()
            hd_values = []
            for batch_idx in range(warped_np.shape[0]):
                labels = np.unique(np.concatenate((warped_np[batch_idx], target_label_np[batch_idx])))
                labels = labels[labels > 0.5]
                for label in labels:
                    hd = compute_hd95(target_label_np[batch_idx, 0] == label, warped_np[batch_idx, 0] == label)
                    if not np.isnan(hd):
                        hd_values.append(hd)
            if hd_values:
                eval_hd95.update(float(np.mean(hd_values)), source.shape[0])
    if compute_extra:
        return eval_loss.avg, 0.0, eval_dsc.avg, eval_hd95.avg, eval_jac.avg, eval_mag.avg
    return eval_loss.avg, 0.0, eval_dsc.avg


def slice_np(tensor, z_idx, is_label=False):
    arr = tensor[:, :, :, :, z_idx].detach().cpu().numpy()[0, 0]
    if is_label:
        arr = np.where(np.isin(arr, VIS_LABELS), arr, 0)
    return np.rot90(arr, -1)


@torch.no_grad()
def save_qualitative_results(model, dataset, output_dir, epoch, device='cuda'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = 9 if len(dataset) > 9 else max(0, len(dataset) // 2)
    source, target, source_label, target_label = dataset[sample_idx]
    source = source.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    source_label = source_label.unsqueeze(0).to(device)
    target_label = target_label.unsqueeze(0).to(device)
    output = model(source, target, return_pos_flow=True)
    moved = output['moved_vol'].float()
    flow = output['pos_flow'].float()
    warped_label = warp_labels(model, source_label, flow)
    source_boundary = boundary_magnitude(source.float())
    target_boundary = boundary_magnitude(target.float())
    moved_boundary = boundary_magnitude(moved.float())
    ring = boundary_ring_mask(foreground_mask(target.float()))

    slice_idx = source.shape[4] // 2
    src = slice_np(source, slice_idx)
    tgt = slice_np(target, slice_idx)
    mov = slice_np(moved, slice_idx)
    src_lbl = slice_np(source_label, slice_idx, is_label=True)
    tgt_lbl = slice_np(target_label, slice_idx, is_label=True)
    wrp_lbl = slice_np(warped_label, slice_idx, is_label=True)
    src_boundary = slice_np(source_boundary, slice_idx)
    tgt_boundary = slice_np(target_boundary, slice_idx)
    mov_boundary = slice_np(moved_boundary, slice_idx)
    ring_slice = slice_np(ring, slice_idx)
    vmax_val = max(np.max(src), np.max(tgt), np.max(mov))
    vmin_val = min(np.min(src), np.min(tgt), np.min(mov))

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for ax in axes.ravel():
        ax.axis('off')
    axes[0, 0].imshow(src, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    axes[0, 0].set_title('Source Image')
    axes[0, 1].imshow(tgt, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    axes[0, 1].set_title('Target Image')
    axes[0, 2].imshow(src - tgt, cmap='bwr', vmin=-1, vmax=1)
    axes[0, 2].set_title('Diff: Source - Target')
    axes[0, 3].imshow(mov - tgt, cmap='bwr', vmin=-1, vmax=1)
    axes[0, 3].set_title('Diff: Deformed - Target')

    def color_for(label, fixed=False):
        label_map = {3: 0, 22: 0, 7: 1, 26: 1, 11: 2, 14: 3, 30: 3}
        pairs = {
            0: ('#1f77b4', '#00bfff'),
            1: ('#ff7f0e', '#ffd700'),
            2: ('#2ca02c', '#32cd32'),
            3: ('#d62728', '#ff69b4'),
        }
        return pairs.get(label_map.get(int(label), int(label)), ('#ffffff', '#ffffff'))[1 if fixed else 0]

    def plot_contours(ax, bg, labels, title, fixed=False):
        ax.imshow(bg, cmap='gray', vmin=vmin_val, vmax=vmax_val)
        for label in np.unique(labels):
            if label > 0:
                mask = labels == label
                if np.any(mask):
                    ax.contour(mask, colors=[color_for(label, fixed=fixed)], linewidths=1.2)
        ax.set_title(title)
        ax.axis('off')

    plot_contours(axes[1, 0], src, src_lbl, 'Source + Labels', fixed=False)
    plot_contours(axes[1, 1], tgt, tgt_lbl, 'Target + Labels', fixed=True)
    plot_contours(axes[1, 2], mov, wrp_lbl, 'Deformed + Labels', fixed=False)
    axes[1, 3].imshow(mov, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    for label in np.unique(np.concatenate([tgt_lbl, wrp_lbl])):
        if label > 0:
            target_mask = tgt_lbl == label
            warped_mask = wrp_lbl == label
            if np.any(target_mask):
                axes[1, 3].contour(target_mask, colors=[color_for(label, fixed=True)], linewidths=1.5, linestyles='dashed', alpha=0.8)
            if np.any(warped_mask):
                axes[1, 3].contour(warped_mask, colors=[color_for(label, fixed=False)], linewidths=1.5)
    axes[1, 3].set_title('Result vs GT')

    disp = flow.detach().cpu().numpy()[0]
    disp_slice = np.rot90(disp[:, :, :, slice_idx], -1, axes=(1, 2))
    disp_slice[1] = -disp_slice[1]
    disp_slice[2] = -disp_slice[2]
    h, w = src.shape
    dx, dy, dz = disp_slice[2], disp_slice[1], disp_slice[0]
    max_mag = np.max(np.abs(disp_slice)) + 1e-5
    flow_vis = np.zeros((h, w, 3), dtype=np.float32)
    flow_vis[..., 0] = (dx / (2 * max_mag)) + 0.5
    flow_vis[..., 1] = (dy / (2 * max_mag)) + 0.5
    flow_vis[..., 2] = (dz / (2 * max_mag)) + 0.5
    axes[2, 0].imshow(np.clip(flow_vis, 0, 1))
    axes[2, 0].set_title('RGB Displacement')
    axes[2, 1].text(0.5, 0.55, f'[{ -max_mag:.2f}, {max_mag:.2f}]', ha='center', va='center', fontsize=16, fontweight='bold')
    axes[2, 1].set_title('Displacement Range')
    grid_spacing = 10
    axes[2, 2].imshow(np.zeros_like(src), cmap='gray', vmin=0, vmax=1)
    for x_idx in range(0, w, grid_spacing):
        if x_idx < disp_slice.shape[2]:
            axes[2, 2].plot(x_idx + disp_slice[2, :, x_idx], np.arange(h) + disp_slice[1, :, x_idx], 'w-', linewidth=0.8, alpha=0.9)
    for y_idx in range(0, h, grid_spacing):
        if y_idx < disp_slice.shape[1]:
            axes[2, 2].plot(np.arange(w) + disp_slice[2, y_idx, :], y_idx + disp_slice[1, y_idx, :], 'w-', linewidth=0.8, alpha=0.9)
    axes[2, 2].set_title('Deformed Grid')
    axes[2, 2].set_ylim(h, 0)
    axes[2, 2].set_xlim(0, w)
    jac = np.rot90(jacobian_determinant(disp)[:, :, slice_idx], -1)
    jac_vis = np.zeros((*jac.shape, 3), dtype=np.float32)
    jac_vis[jac < 0] = (1.0, 0.0, 0.0)
    jac_vis[(jac >= 0) & (jac <= 1)] = (0.4, 0.8, 0.4)
    jac_vis[jac > 1] = (0.4, 0.6, 0.9)
    axes[2, 3].imshow(jac_vis)
    axes[2, 3].set_title('Jacobian Determinant')

    boundary_vmax = max(np.max(src_boundary), np.max(tgt_boundary), np.max(mov_boundary), 1e-6)
    axes[3, 0].imshow(src_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
    axes[3, 0].set_title('Source Boundary Map')
    axes[3, 1].imshow(tgt_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
    axes[3, 1].set_title('Target Boundary Map')
    axes[3, 2].imshow(mov_boundary, cmap='magma', vmin=0.0, vmax=boundary_vmax)
    axes[3, 2].set_title('Warped Boundary Map')
    axes[3, 3].imshow(tgt, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    axes[3, 3].imshow(ring_slice, cmap='autumn', alpha=0.45, vmin=0.0, vmax=1.0)
    axes[3, 3].set_title('Boundary Ring Mask')

    plt.suptitle(f'Epoch {epoch} - Sample default (Slice Z={slice_idx})', fontsize=16)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except UserWarning:
        pass
    plt.savefig(output_dir / f'vis_epoch_{epoch:04d}_default.png')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    parser.add_argument('--output', default='/root/autodl-tmp/models/oasis_mambamorph.pt')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', choices=['mse', 'ncc'], default='ncc')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.01)
    parser.add_argument('--integration-steps', type=int, choices=[5, 7], default=5, help='Scaling-and-squaring integration steps for MambaMorph')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--vis-every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--warm-start', type=int, default=10)
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--remark',
        default='MambaMorph comparison model on OASIS; unsupervised diffeomorphic registration with 5-step integration; NCC image loss; batch size 2; data loading, main hyperparameters, metrics, visualization, and logging aligned with /Users/yiang/zc/voxelmorph/scripts/train_oasis.py.',
    )
    return parser.parse_args()


def write_config(path, args, model, device, output):
    with path.open('w') as handle:
        handle.write('Training Configuration:\n')
        handle.write(f'Device: {device}\n')
        handle.write(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n')
        handle.write(f'Epochs: {args.epochs}\n')
        handle.write(f'Start Epoch: {args.start_epoch}\n')
        handle.write(f'Resume: {args.resume}\n')
        handle.write(f'Seed: {args.seed}\n')
        handle.write(f'Batch Size: {args.batch_size}\n')
        handle.write(f'Accumulation Steps: {args.accumulation_steps}\n')
        handle.write(f'Lambda: {args.lambda_param}\n')
        handle.write(f'LR: {args.lr}\n')
        handle.write(f'Loss: {args.loss}\n')
        handle.write(f'Use Mask: {args.use_mask}\n')
        handle.write(f'Integration Steps: {args.integration_steps}\n')
        handle.write(f'Warmup Epochs: {args.warmup_epochs}\n')
        handle.write('Model Architecture: MambaMorph\n')
        handle.write(f'Train Dir: {args.train_dir}\n')
        handle.write(f'Val Dir: {args.val_dir}\n')
        handle.write(f'Output Path: {output}\n')
        handle.write(f'Command: {" ".join(shlex.quote(arg) for arg in sys.argv)}\n')
        handle.write(f'Remark: {args.remark}\n')
        handle.write(f'Arguments: {vars(args)}\n')


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = OASISTrainDataset(glob.glob(os.path.join(args.train_dir, '*.pkl')))
    val_set = OASISValidationDataset(glob.glob(os.path.join(args.val_dir, '*.pkl')))
    inshape = tuple(train_set[0][0].shape[1:])
    model = build_model(inshape, device, args.resume, integration_steps=args.integration_steps)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == 'cuda', generator=train_generator)
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=args.workers, pin_memory=device == 'cuda', drop_last=True)
    image_loss = make_image_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    if warmup_epochs:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs)),
            ],
            milestones=[warmup_epochs],
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

    beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    requested_output = Path(args.output)
    run_dir = requested_output.parent / f'{requested_output.stem}_{timestamp}'
    output = run_dir / requested_output.name
    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = output.parent / 'train_log.csv'
    config_file = output.parent / 'config.txt'
    write_config(config_file, args, model, device, output)
    with log_file.open('w', newline='') as handle:
        csv.writer(handle).writerow(['epoch', 'train_loss', 'train_img_loss', 'train_grad_loss', 'train_feature_edge_loss', 'val_dsc', 'val_hd95', 'val_jac', 'val_mag'])

    print(f'MambaMorph parameters: {sum(p.numel() for p in model.parameters()):,}; inshape: {inshape}; device: {device}')
    print(f'Output directory for this run: {output.parent}')
    print(f'Training for {args.epochs} epochs...')
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    best_dsc = 0.0
    loss_history = []
    val_dsc_history = []
    epoch_times = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_img, train_grad = train_epoch(
            model, train_loader, optimizer, scaler, image_loss, args.loss, args.lambda_param, device, amp, args.use_mask, args.accumulation_steps
        )
        loss_history.append(float(train_loss))
        _, _, val_dsc = validate(model, val_loader, image_loss, args.loss, args.lambda_param, device, compute_extra=False, use_mask=args.use_mask)
        is_new_best = val_dsc > best_dsc
        compute_extra = epoch in [1, 3, 5, 7] or epoch % 10 == 0 or (is_new_best and val_dsc > 0.77)
        if compute_extra:
            _, _, _, val_hd95, val_jac, val_mag = validate(
                model, val_loader, image_loss, args.loss, args.lambda_param, device, compute_extra=True, use_mask=args.use_mask
            )
        else:
            val_hd95 = val_jac = val_mag = np.nan
        val_dsc_history.append(float(val_dsc))
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        if compute_extra:
            print(f'Epoch {epoch}/{args.epochs}, Loss: {train_loss:.6f}, Img: {train_img:.6f}, Grad: {train_grad:.6f}, FeatureEdge: 0.000000, Val DSC: {val_dsc:.6f}, HD95: {val_hd95:.2f}, Jac: {val_jac:.4f}, Mag: {val_mag:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')
        else:
            print(f'Epoch {epoch}/{args.epochs}, Loss: {train_loss:.6f}, Img: {train_img:.6f}, Grad: {train_grad:.6f}, FeatureEdge: 0.000000, Val DSC: {val_dsc:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Peak: {peak_gpu_mem:.2f}MB')
        if compute_extra or (args.vis_every and epoch % args.vis_every == 0):
            try:
                save_qualitative_results(model, val_set, output.parent, epoch=epoch, device=device)
            except Exception as exc:
                print(f'Failed to save visualization: {exc}')
        with log_file.open('a', newline='') as handle:
            csv.writer(handle).writerow([
                epoch,
                f'{train_loss:.6f}',
                f'{train_img:.6f}',
                f'{train_grad:.6f}',
                '0.000000',
                f'{val_dsc:.6f}',
                f'{val_hd95:.2f}' if compute_extra else '',
                f'{val_jac:.6f}' if compute_extra else '',
                f'{val_mag:.6f}' if compute_extra else '',
            ])
        if len(loss_history) >= args.warm_start + args.patience + 1:
            recent = loss_history[-args.patience:]
            best_past = min(loss_history[:-args.patience])
            if all(max(best_past - value, 0) < args.threshold for value in recent):
                print(f'Early stopping at epoch {epoch}')
                break
        if epoch % args.save_every == 0:
            checkpoint_path = output.parent / f'{output.stem}_epoch{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
        if is_new_best:
            best_dsc = val_dsc
            best_path = output.parent / f'{output.stem}_best.pt'
            torch.save(model.state_dict(), best_path)
            print(f'Saved new best model with DSC: {best_dsc:.6f} (HD95: {val_hd95:.2f}, Jac: {val_jac:.4f})')
        if epoch % 10 == 0:
            try:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Train Loss', color='tab:red')
                ax1.plot(range(1, len(loss_history) + 1), loss_history, color='tab:red', marker='o', markersize=4, label='Train Loss')
                ax1.tick_params(axis='y', labelcolor='tab:red')
                ax2 = ax1.twinx()
                ax2.set_ylabel('Val DSC', color='tab:blue')
                ax2.plot(range(1, len(val_dsc_history) + 1), val_dsc_history, color='tab:blue', marker='s', markersize=4, label='Val DSC')
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left')
                plt.title(f'Learning Curves (Epoch 1 to {epoch})')
                fig.tight_layout()
                ax1.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(output.parent / f'learning_curves_epoch{epoch}.png', dpi=150)
                plt.close(fig)
            except Exception as exc:
                print(f'Failed to save learning curve plot: {exc}')

    torch.save(model.state_dict(), output)
    print(f'Final model saved to {output}')
    finish_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    with config_file.open('a') as handle:
        handle.write(f'End Time: {finish_time.strftime("%Y%m%d_%H%M%S")}\n')
        handle.write(f'Average Epoch Time: {sum(epoch_times) / len(epoch_times):.2f} s\n')
        final_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device == 'cuda' else 0.0
        handle.write(f'Peak GPU Memory: {final_peak_mem:.2f} MB\n')


if __name__ == '__main__':
    main()
