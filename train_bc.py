#!/usr/bin/env python3
"""
Behavior Cloning training on derived dataset.

- Input: bc_dataset.npz with obs[N,10], act[N,3] (steer, throttle, brake)
- Model: small MLP; saves bc_policy.pt

Usage:
  python train_bc.py --data bc_dataset_run1.npz --epochs 50 --out bc_policy.pt
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    def __init__(self, in_dim=10, out_dim=3, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]  # steer in [-1,1]; we'll clamp throttle/brake later
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        # map tanh outputs to desired ranges: steer [-1,1], throttle/brake [0,1]
        steer = y[:, 0:1]
        tb = (y[:, 1:] + 1.0) * 0.5  # [0,1]
        return torch.cat([steer, tb], dim=1)


def train(data_path: str, out_path: str, epochs=50, batch=2048, lr=1e-3, val_split=0.1, device='cpu'):
    data = np.load(data_path)
    obs = torch.tensor(data['obs'], dtype=torch.float32)
    act = torch.tensor(data['act'], dtype=torch.float32)

    dataset = TensorDataset(obs, act)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    model = MLP().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # Losses: MSE + small smoothness on steer via Δ between consecutive preds
    mse = nn.MSELoss()

    def run_epoch(dl, train=True):
        model.train(train)
        total, n = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = mse(pred, yb)
            # In-batch steer smoothness (no cross-batch dependency)
            if pred.shape[0] > 1:
                smooth = ((pred[1:, 0] - pred[:-1, 0])**2).mean()
                loss = loss + 0.001 * smooth
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item() * len(xb); n += len(xb)
        return total / max(1, n)


    train_dl = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_set, batch_size=batch, shuffle=False, drop_last=False)

    for ep in range(1, epochs+1):
        tr = run_epoch(train_dl, True)
        va = run_epoch(val_dl, False)
        print(f"epoch {ep:03d}  train {tr:.5f}  val {va:.5f}")

    torch.save(model.state_dict(), out_path)
    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='bc_policy.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(args.data, args.out, epochs=args.epochs, batch=args.batch, lr=args.lr, device=device)


if __name__ == '__main__':
    main()
