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


def is_nifti_file(path):
    return path.is_file() and path.name.endswith(('.nii', '.nii.gz'))


def nifti_files(directory):
    if directory is None:
        return []
    path = Path(directory)
    return sorted([item for item in path.iterdir() if is_nifti_file(item)]) if path.exists() else []


def discover_mask_dir(image_dir):
    if image_dir is None:
        return None
    image_dir = Path(image_dir)
    mask_dir = image_dir.parent.parent / 'masks' / image_dir.name
    return mask_dir if mask_dir.exists() else None


def case_id(path):
    stem = path.name.removesuffix('.gz').removesuffix('.nii')
    return stem[:-5] if stem.endswith(('_0000', '_0001')) else stem


def load_volume(path, label=False, binarize=False):
    array = nib.load(str(path)).get_fdata().astype(np.float32)
    if label:
        return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
    if binarize:
        array = (array > 0.5).astype(np.float32)
    else:
        array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)


def mask_map(image_dir):
    mask_dir = discover_mask_dir(image_dir)
    if not mask_dir:
        return {}
    return {path.name: path for path in nifti_files(mask_dir)}


class MultimodalTrainDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, paired_ct_dir=None, paired_mr_dir=None, max_samples=None, unpaired=False):
        self.ct_dir, self.mr_dir = Path(ct_dir), Path(mr_dir)
        self.ct, self.mr = nifti_files(ct_dir), nifti_files(mr_dir)
        self.ct_masks, self.mr_masks = mask_map(ct_dir), mask_map(mr_dir)
        self.cache = {}
        if not self.ct or not self.mr:
            raise ValueError('Both CT and MR training directories must contain NIfTI files.')
        paired_ct, paired_mr = nifti_files(paired_ct_dir), nifti_files(paired_mr_dir)
        self.paired_ct_masks, self.paired_mr_masks = mask_map(paired_ct_dir), mask_map(paired_mr_dir)
        mr_map = {case_id(path): path for path in paired_mr}
        self.pairs = [(path, mr_map[case_id(path)]) for path in paired_ct if case_id(path) in mr_map]
        self.random_samples = max_samples if max_samples is not None else 100
        self.unpaired = unpaired
        print(f'Training Dataset: {len(self.ct)} Unpaired CTs, {len(self.mr)} Unpaired MRs.')
        print(f'                  + {len(self.pairs)} Fixed Pairs found in {paired_ct_dir if paired_ct_dir else "None"}')
        print(f'                  Epoch Length: {self.random_samples} Random + {len(self.pairs)} Fixed = {len(self)}')

    def __len__(self):
        return self.random_samples + len(self.pairs)

    def _load(self, path, label=False, binarize=False):
        prefix = 'label' if label else ('mask' if binarize else 'image')
        key = f'{prefix}::{path}'
        if key not in self.cache:
            self.cache[key] = load_volume(path, label=label, binarize=binarize)
        return self.cache[key]

    def __getitem__(self, index):
        if index < self.random_samples:
            source = self.ct[torch.randint(len(self.ct), ()).item()]
            target = self.mr[torch.randint(len(self.mr), ()).item()]
            source_masks, target_masks = self.ct_masks, self.mr_masks
        else:
            source, target = self.pairs[index - self.random_samples]
            source_masks, target_masks = self.paired_ct_masks, self.paired_mr_masks
        sample = {'source': self._load(source), 'target': self._load(target)}
        if source.name in source_masks:
            sample['source_mask'] = self._load(source_masks[source.name], binarize=True)
        if target.name in target_masks:
            sample['target_mask'] = self._load(target_masks[target.name], binarize=True)
        return sample


class MultimodalValidationDataset(Dataset):
    def __init__(self, ct_dir, mr_dir, ct_label_dir=None, mr_label_dir=None, paired=False):
        self.ct_paths, self.mr_paths = nifti_files(ct_dir), nifti_files(mr_dir)
        self.ct_masks, self.mr_masks = mask_map(ct_dir), mask_map(mr_dir)
        self.ct_labels = {path.name: path for path in nifti_files(ct_label_dir)}
        self.mr_labels = {path.name: path for path in nifti_files(mr_label_dir)}
        self.cache = {}
        self.lbl_cache = {}
        self.pairs = []
        if paired:
            mr_map = {case_id(path): path for path in self.mr_paths}
            self.pairs = [(path, mr_map[case_id(path)]) for path in self.ct_paths if case_id(path) in mr_map]
        else:
            self.pairs = [(ct, mr) for ct in self.ct_paths for mr in self.mr_paths]
        print(f'Validation/Test Dataset: {len(self.ct_paths)} CTs, {len(self.mr_paths)} MRs, {len(self.pairs)} pairs (paired={paired}).')

    def __len__(self):
        return len(self.pairs)

    def _load(self, path, label=False, binarize=False):
        cache = self.lbl_cache if label else self.cache
        prefix = 'label' if label else ('mask' if binarize else 'image')
        key = f'{prefix}::{path}'
        if key not in cache:
            cache[key] = load_volume(path, label=label, binarize=binarize)
        return cache[key]

    def __getitem__(self, index):
        source, target = self.pairs[index]
        sample = {'source': self._load(source), 'target': self._load(target)}
        if source.name in self.ct_masks:
            sample['source_mask'] = self._load(self.ct_masks[source.name], binarize=True)
        if target.name in self.mr_masks:
            sample['target_mask'] = self._load(self.mr_masks[target.name], binarize=True)
        if source.name in self.ct_labels:
            sample['source_label'] = self._load(self.ct_labels[source.name], label=True)
        if target.name in self.mr_labels:
            sample['target_label'] = self._load(self.mr_labels[target.name], label=True)
        sample['filename'] = source.name.removesuffix('.gz').removesuffix('.nii')
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
    parser.add_argument('--ct-test-dir', default=f'{root}/test/images/ct')
    parser.add_argument('--mr-test-dir', default=f'{root}/test/images/mr')
    parser.add_argument('--ct-test-label-dir', default=f'{root}/test/labels/ct')
    parser.add_argument('--mr-test-label-dir', default=f'{root}/test/labels/mr')
    parser.add_argument('--output', default='/root/autodl-tmp/models/multimodal_mambamorph.pt')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
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
    parser.add_argument('--unpaired', action='store_true', default=False)
    parser.add_argument('--val-paired', action='store_true', default=False)
    parser.add_argument('--test-paired', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--warm-start', type=int, default=10)
    parser.add_argument('--visualize-every', type=int, default=0, help='Extra visualization interval; 0 follows the reference test-monitor schedule.')
    parser.add_argument(
        '--remark',
        default='MambaMorph comparison model; CT-to-MR multimodal registration; unsupervised diffeomorphic training; MI image loss; logs/metrics/visualization aligned with Voxelmorph train_multimodal.py.',
        help='Short note written into config.txt to summarize the core purpose/settings of this run.',
    )
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
    train_set = MultimodalTrainDataset(args.ct_dir, args.mr_dir, args.paired_ct_dir, args.paired_mr_dir, args.max_train_samples, args.unpaired)
    val_set = MultimodalValidationDataset(args.ct_val_dir, args.mr_val_dir, args.ct_val_label_dir, args.mr_val_label_dir, paired=args.val_paired)
    test_set = MultimodalValidationDataset(args.ct_test_dir, args.mr_test_dir, args.ct_test_label_dir, args.mr_test_label_dir, paired=args.test_paired)
    inshape = tuple(train_set[0]['source'].shape[1:])
    model = build_model(inshape, device, args.resume)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == 'cuda', persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == 'cuda') if len(val_set) else None
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == 'cuda') if len(test_set) else None
    image_loss = make_image_loss(args.loss, args.ncc_win, args.mi_bins)
    print(f'MambaMorph parameters: {sum(p.numel() for p in model.parameters()):,}; inshape: {inshape}; device: {device}')
    run_training(args, model, train_loader, val_loader, test_loader, image_loss, device)


if __name__ == '__main__':
    main()
