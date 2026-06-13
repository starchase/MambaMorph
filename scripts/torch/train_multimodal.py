#!/usr/bin/env python3
"""Unsupervised diffeomorphic MambaMorph training for CT-to-MR registration."""

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mambamorph_train_common import build_model, make_image_loss, run_training, seed_everything


def nifti_files(directory):
    path = Path(directory)
    return sorted([item for item in path.iterdir() if item.name.endswith(('.nii', '.nii.gz'))]) if path.exists() else []


def case_id(path):
    stem = path.name.removesuffix('.gz').removesuffix('.nii')
    return stem[:-5] if stem.endswith(('_0000', '_0001')) else stem


def load_volume(path, label=False):
    array = nib.load(str(path)).get_fdata()
    if label:
        return torch.from_numpy(np.ascontiguousarray(array)).long().unsqueeze(0)
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, (1, 99))
    array = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)


def mask_map(image_dir):
    if not image_dir:
        return {}
    image_dir = Path(image_dir)
    return {case_id(path): path for path in nifti_files(image_dir.parent.parent / 'masks' / image_dir.name)}


class MultimodalTrainDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, paired_ct_dir=None, paired_mr_dir=None, max_samples=None):
        self.ct, self.mr = nifti_files(ct_dir), nifti_files(mr_dir)
        self.ct_masks, self.mr_masks = mask_map(ct_dir), mask_map(mr_dir)
        if not self.ct or not self.mr:
            raise ValueError('Both CT and MR training directories must contain NIfTI files.')
        paired_ct, paired_mr = nifti_files(paired_ct_dir), nifti_files(paired_mr_dir)
        self.paired_ct_masks, self.paired_mr_masks = mask_map(paired_ct_dir), mask_map(paired_mr_dir)
        mr_map = {case_id(path): path for path in paired_mr}
        self.pairs = [(path, mr_map[case_id(path)]) for path in paired_ct if case_id(path) in mr_map]
        self.random_samples = max_samples if max_samples is not None else 100

    def __len__(self):
        return self.random_samples + len(self.pairs)

    def __getitem__(self, index):
        if index < self.random_samples:
            source = self.ct[torch.randint(len(self.ct), ()).item()]
            target = self.mr[torch.randint(len(self.mr), ()).item()]
            target_masks = self.mr_masks
        else:
            source, target = self.pairs[index - self.random_samples]
            target_masks = self.paired_mr_masks
        sample = {'source': load_volume(source), 'target': load_volume(target)}
        if case_id(target) in target_masks:
            sample['target_mask'] = (load_volume(target_masks[case_id(target)]) > 0.5).float()
        return sample


class MultimodalValidationDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, ct_label_dir=None, mr_label_dir=None):
        ct_map, mr_map = ({case_id(path): path for path in nifti_files(directory)} for directory in (ct_dir, mr_dir))
        self.pairs = [(ct_map[key], mr_map[key], key) for key in sorted(ct_map.keys() & mr_map.keys())]
        self.ct_labels = {case_id(path): path for path in nifti_files(ct_label_dir)}
        self.mr_labels = {case_id(path): path for path in nifti_files(mr_label_dir)}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target, key = self.pairs[index]
        sample = {'source': load_volume(source), 'target': load_volume(target)}
        if key in self.ct_labels and key in self.mr_labels:
            sample.update(source_label=load_volume(self.ct_labels[key], True), target_label=load_volume(self.mr_labels[key], True))
        return sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = '/root/autodl-tmp/classedAbdomenMRCT_norm_300'
    parser.add_argument('--ct-dir', default=f'{root}/train/images/ct')
    parser.add_argument('--mr-dir', default=f'{root}/train/images/mr')
    parser.add_argument('--paired-ct-dir', default=f'{root}/trainPairs/images/ct')
    parser.add_argument('--paired-mr-dir', default=f'{root}/trainPairs/images/mr')
    parser.add_argument('--ct-val-dir', default=f'{root}/val/images/ct')
    parser.add_argument('--mr-val-dir', default=f'{root}/val/images/mr')
    parser.add_argument('--ct-val-label-dir', default=f'{root}/val/labels/ct')
    parser.add_argument('--mr-val-label-dir', default=f'{root}/val/labels/mr')
    parser.add_argument('--output', default='/root/autodl-tmp/models/multimodal_mambamorph.pt')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.01)
    parser.add_argument('--integration-steps', type=int, choices=[7], default=7, help='MambaMorph uses fixed 7-step scaling-and-squaring')
    parser.add_argument('--image-loss', dest='loss', choices=['mse', 'ncc', 'mi'], default='ncc')
    parser.add_argument('--ncc-win', type=int, default=9)
    parser.add_argument('--mi-bins', type=int, default=32)
    parser.add_argument('--loss-mask-mode', choices=['manual', 'auto', 'none'], default='manual')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = MultimodalTrainDataset(args.ct_dir, args.mr_dir, args.paired_ct_dir, args.paired_mr_dir, args.max_train_samples)
    val_set = MultimodalValidationDataset(args.ct_val_dir, args.mr_val_dir, args.ct_val_label_dir, args.mr_val_label_dir)
    inshape = tuple(train_set[0]['source'].shape[1:])
    model = build_model(inshape, device, args.resume)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == 'cuda', persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == 'cuda') if len(val_set) else None
    image_loss = make_image_loss(args.loss, args.ncc_win, args.mi_bins)
    print(f'MambaMorph parameters: {sum(p.numel() for p in model.parameters()):,}; inshape: {inshape}; device: {device}')
    run_training(args, model, train_loader, val_loader, image_loss, device)


if __name__ == '__main__':
    main()
