#!/usr/bin/env python3
"""
Derive a Behavior Cloning dataset (obs, action) from recorded telemetry by estimating
actions from kinematics (curvature -> steer, speed change -> throttle/brake).

This is an approximation useful when we didn't log human inputs. Good enough to warm-start.

Input: one or more .npz files produced by tmnf_record_telemetry.py
Output: bc_dataset.npz with arrays: obs[N,10], act[N,3] (steer, throttle, brake)

Usage:
  python tmnf_derive_bc_dataset.py --out bc_dataset_run1.npz laps_run1.npz laps_run2.npz
"""
from __future__ import annotations
import argparse
import numpy as np

# Hyper-params for derivation
STEER_GAIN = 0.8      # map curvature to steer
STEER_SAT  = 1.0
ACCEL_GAIN = 0.15     # maps dv/dt (m/s^2) to throttle
BRAKE_GAIN = 0.25     # maps negative dv/dt to brake
SMOOTH_W   = 5        # window for smoothing derivatives


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < 2*w+1:
        return x
    kernel = np.ones(2*w+1) / float(2*w+1)
    return np.convolve(x, kernel, mode='same')


def derive_actions(obs: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    """
    Safer derivation that:
      - splits on time resets (non-increasing t_ms),
      - uses finite differences with clamped dt,
      - guards against near-zero speed when computing curvature.
    """
    # columns: [speed_n, yaw_s, yaw_c, pitch, roll, v_fwd_n, v_lat_n, on_ground, cp_scaled, t_scaled]
    yaw = np.arctan2(obs[:, 1], obs[:, 2])
    speed_mps = np.clip(obs[:, 0], 0, 1) * (300.0 / 3.6)

    # --- split into monotonic time segments ---
    t_s = t_ms.astype(np.float64) / 1000.0
    breaks = np.where(np.diff(t_s) <= 0)[0] + 1  # indices where time resets
    segments = np.split(np.arange(len(t_s)), breaks)

    steer_all, throttle_all, brake_all = [], [], []
    eps_dt = 1e-3   # min dt (s)
    min_speed = 0.5 # m/s for curvature denom

    for seg in segments:
        if seg.size < 3:
            # too short to get good derivatives; fill zeros
            steer_all.append(np.zeros(seg.size, dtype=np.float32))
            throttle_all.append(np.zeros(seg.size, dtype=np.float32))
            brake_all.append(np.zeros(seg.size, dtype=np.float32))
            continue

        idx = seg
        dt = np.diff(t_s[idx])
        dt = np.maximum(dt, eps_dt)

        # yaw rate
        dyaw = np.diff(yaw[idx])
        dyaw_dt = np.empty_like(yaw[idx], dtype=np.float64)
        dyaw_dt[0] = dyaw[0] / dt[0]
        dyaw_dt[1:-1] = (yaw[idx][2:] - yaw[idx][:-2]) / np.maximum(t_s[idx][2:] - t_s[idx][:-2], eps_dt)
        dyaw_dt[-1] = dyaw[-1] / dt[-1]

        # curvature ~ yaw_rate / speed
        sp = np.maximum(speed_mps[idx], min_speed)
        kappa = dyaw_dt / sp

        steer = np.clip(smooth(STEER_GAIN * kappa, SMOOTH_W), -STEER_SAT, STEER_SAT)

        # accel from speed derivative
        dv = np.diff(speed_mps[idx])
        accel = np.empty_like(speed_mps[idx])
        accel[0] = dv[0] / dt[0]
        accel[1:-1] = (speed_mps[idx][2:] - speed_mps[idx][:-2]) / np.maximum(t_s[idx][2:] - t_s[idx][:-2], eps_dt)
        accel[-1] = dv[-1] / dt[-1]
        accel = smooth(accel, SMOOTH_W)

        throttle = np.clip(ACCEL_GAIN * np.maximum(0.0, accel), 0.0, 1.0)
        brake    = np.clip(BRAKE_GAIN * np.maximum(0.0, -accel), 0.0, 1.0)

        steer_all.append(steer.astype(np.float32))
        throttle_all.append(throttle.astype(np.float32))
        brake_all.append(brake.astype(np.float32))

    steer = np.concatenate(steer_all, axis=0)
    throttle = np.concatenate(throttle_all, axis=0)
    brake = np.concatenate(brake_all, axis=0)
    return np.stack([steer, throttle, brake], axis=1).astype(np.float32)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('inputs', nargs='+')
    args = ap.parse_args()

    obs_list, act_list = [], []
    for path in args.inputs:
        data = np.load(path, allow_pickle=True)
        obs = data['obs']
        t_ms = data['t_ms']
        act = derive_actions(obs, t_ms)
        obs_list.append(obs)
        act_list.append(act)

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)

    np.savez_compressed(args.out, obs=obs, act=act)
    print(f"Saved BC dataset: {obs.shape[0]} samples -> {args.out}")


if __name__ == '__main__':
    main()
