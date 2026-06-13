#!/usr/bin/env python3
"""Unsupervised diffeomorphic MambaMorph training on OASIS."""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mambamorph_train_common import build_model, make_image_loss, run_training, seed_everything


class OASISTrainDataset(Dataset):
    def __init__(self, paths):
        self.paths = sorted(paths)
        if len(self.paths) < 2:
            raise ValueError('OASIS training requires at least two .pkl files.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        target_index = torch.randint(0, len(self.paths) - 1, ()).item()
        target_index += target_index >= index
        source, _ = pickle.load(open(self.paths[index], 'rb'))
        target, _ = pickle.load(open(self.paths[target_index], 'rb'))
        return {'source': volume(source), 'target': volume(target)}


class OASISValidationDataset(Dataset):
    def __init__(self, paths):
        self.paths = sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        source, target, source_label, target_label = pickle.load(open(self.paths[index], 'rb'))
        return {
            'source': volume(source),
            'target': volume(target),
            'source_label': volume(source_label, torch.float32),
            'target_label': volume(target_label, torch.int64),
        }


def volume(array, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', default='/root/autodl-tmp/OASIS_L2R_2021_task03/All/')
    parser.add_argument('--val-dir', default='/root/autodl-tmp/OASIS_L2R_2021_task03/Test/')
    parser.add_argument('--output', default='/root/autodl-tmp/models/oasis_mambamorph.pt')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', choices=['mse', 'ncc'], default='mse')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.01)
    parser.add_argument('--integration-steps', type=int, choices=[7], default=7, help='MambaMorph uses fixed 7-step scaling-and-squaring')
    parser.add_argument('--loss-mask-mode', choices=['none', 'auto'], default='none')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = OASISTrainDataset(glob.glob(os.path.join(args.train_dir, '*.pkl')))
    val_set = OASISValidationDataset(glob.glob(os.path.join(args.val_dir, '*.pkl')))
    inshape = tuple(train_set[0]['source'].shape[1:])
    model = build_model(inshape, device, args.resume)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == 'cuda')
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=args.workers, pin_memory=device == 'cuda')
    print(f'MambaMorph parameters: {sum(p.numel() for p in model.parameters()):,}; inshape: {inshape}; device: {device}')
    run_training(args, model, train_loader, val_loader, make_image_loss(args.loss), device)


if __name__ == '__main__':
    main()
