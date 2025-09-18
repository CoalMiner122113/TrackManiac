#!/usr/bin/env python3
"""
PPO fine-tuning starting from a Behavior Cloning policy — with an actual warm start.

We do a short **supervised warm-up** inside the SB3 policy before RL, using your BC dataset
(so no brittle weight-copying across frameworks). The warm-up trains the policy's action
mean to match the BC actions for the first 3 dims [steer, throttle, brake]. The remaining
2 dims [air_pitch, air_roll] are initialized to zero.

Usage:
  python ppo_finetune_from_bc.py --bc_data bc_dataset_run1.npz \
      --warmup_epochs 5 --timesteps 500000
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tmnf_gym_env import TMNFEnv, EnvConfig


# ---------------- Supervised warm-up -----------------
@torch.no_grad()
def _zero_air_dims(policy) -> None:
    """Initialize air control (last 2 action dims) to near-zero mean."""
    if hasattr(policy, 'action_net'):
        # action_net: Linear(last_pi_dim -> action_dim)
        w = policy.action_net.weight.data  # [act_dim, hid]
        b = policy.action_net.bias.data    # [act_dim]
        # Zero rows for dims 3 and 4 (air_pitch, air_roll)
        if b.shape[0] >= 5:
            w[3:].zero_()
            b[3:].zero_()


def supervised_warmup(policy, venv_norm: Optional[VecNormalize], bc_npz: str, epochs: int = 5,
                      batch_size: int = 2048, device: str = 'cpu') -> None:
    """Train the SB3 policy mean to match BC actions (steer/throttle/brake) for a few epochs.
    Only updates the actor head and (if present) the policy-specific MLP; leaves value net frozen.
    Compatible with recent SB3 where ActorCriticPolicy has mlp_extractor & action_net, but no `.actor` attr.
    """
    data = np.load(bc_npz)
    obs = torch.tensor(data['obs'], dtype=torch.float32, device=device)
    act = torch.tensor(data['act'], dtype=torch.float32, device=device)  # [N,3]

    ds = TensorDataset(obs, act)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # Collect trainable params robustly across SB3 versions
    train_params = []
    if hasattr(policy, 'action_net'):
        train_params += list(policy.action_net.parameters())
    # policy.mlp_extractor has policy_net when separate pi/vf; fall back to shared_net otherwise
    if hasattr(policy, 'mlp_extractor'):
        me = policy.mlp_extractor
        if hasattr(me, 'policy_net') and getattr(me, 'policy_net') is not None:
            train_params += list(me.policy_net.parameters())
        elif hasattr(me, 'shared_net') and getattr(me, 'shared_net') is not None:
            train_params += list(me.shared_net.parameters())

    if not train_params:
        raise RuntimeError("Could not find policy parameters to warm up; SB3 policy structure unexpected.")

    opt = torch.optim.Adam(train_params, lr=3e-4)
    mse = nn.MSELoss()

    policy.train(True)

    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for xb, yb in dl:
            # Normalize obs if VecNormalize exists
            if venv_norm is not None:
                xb_np = xb.detach().cpu().numpy()
                xb_np = venv_norm.normalize_obs(xb_np)
                xb = torch.tensor(xb_np, dtype=torch.float32, device=device)

            dist = policy.get_distribution(xb)
            mu = dist.distribution.mean  # [B, act_dim]
            loss = mse(mu[:, :3], yb)

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.shape[0]; n += xb.shape[0]
        print(f"[BC warmup] epoch {ep}/{epochs} loss={total/max(1,n):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc_data', required=True, help='path to bc_dataset .npz for supervised warm-up')
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--timesteps', type=int, default=500_000)
    ap.add_argument('--frame_skip', type=int, default=4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_env():
        return TMNFEnv(EnvConfig(frame_skip=args.frame_skip))

    # Single-env + normalization
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Match policy net to BC capacity (128,128) for a closer fit
    policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        "MlpPolicy", venv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=4,
        gamma=0.995, gae_lambda=0.95, ent_coef=0.003, clip_range=0.2, verbose=1,
        device=device,
    )

    # Initialize air dims to zero-ish so they don't interfere
    _zero_air_dims(model.policy)

    # Supervised warm-up on BC dataset (policy only)
    supervised_warmup(model.policy, venv if isinstance(venv, VecNormalize) else None,
                      args.bc_data, epochs=args.warmup_epochs, device=device)

    # Now run RL fine-tuning
    model.learn(total_timesteps=args.timesteps)
    model.save('tmnf_ppo_finetuned.zip')
    venv.save('tmnf_vecnorm.pkl')


if __name__ == '__main__':
    main()
