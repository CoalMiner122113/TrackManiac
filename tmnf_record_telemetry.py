#!/usr/bin/env python3
"""
Record telemetry from RLBridge into an .npz file at a downsampled rate (default 15 Hz).
We derive observations identical to TMNFEnv._obs_from so you can later train BC or compute pseudo-actions.

Usage:
  python tmnf_record_telemetry.py --out laps_run1.npz --seconds 900

Press Ctrl+C to stop early.
"""
from __future__ import annotations
import argparse
import math
import signal
import time
from typing import List

import numpy as np

from tmnf_rl_bridge_client import RLBridgeClient, Telemetry

MAX_SPEED_KMH = 300.0


def normalize_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def obs_from(fr: Telemetry, max_episode_time: float = 90.0) -> np.ndarray:
    yaw = normalize_angle(fr.yaw)
    pitch = float(np.clip(fr.pitch / 0.6, -1.0, 1.0))
    roll = float(np.clip(fr.roll / 0.8, -1.0, 1.0))

    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, vz = fr.vel
    v_fwd = c * vx + s * vz
    v_lat = -s * vx + c * vz
    v_fwd_n = float(np.tanh(v_fwd / 50.0))
    v_lat_n = float(np.tanh(v_lat / 50.0))

    speed_n = float(np.clip(fr.speed_kmh / MAX_SPEED_KMH, 0.0, 1.0))
    yaw_s, yaw_c = math.sin(yaw), math.cos(yaw)
    on_ground = 1.0 if fr.on_ground else 0.0
    cp_scaled = float(np.tanh(fr.checkpoint / 10.0))
    t_scaled = float(np.clip((fr.race_time_ms / 1000.0) / max_episode_time, 0.0, 1.0))

    return np.array([speed_n, yaw_s, yaw_c, pitch, roll, v_fwd_n, v_lat_n,
                     on_ground, cp_scaled, t_scaled], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5555)
    ap.add_argument('--hz', type=float, default=15.0)
    ap.add_argument('--seconds', type=float, default=600.0, help='max duration to record')
    ap.add_argument('--out', required=True, help='output npz file path')
    args = ap.parse_args()

    period = 1.0 / max(1e-3, args.hz)
    obs_list: List[np.ndarray] = []
    t_ms_list: List[int] = []
    cp_list: List[int] = []

    stop = False
    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    with RLBridgeClient(args.host, args.port, timeout=1.0) as cli:
        print('Recording... (Ctrl+C to stop)')
        t0 = time.time()
        next_t = t0
        while not stop and (time.time() - t0) < args.seconds:
            fr = cli.recv_frame(block=True)
            if fr is None:
                continue
            o = obs_from(fr)
            obs_list.append(o)
            t_ms_list.append(fr.race_time_ms)
            cp_list.append(fr.checkpoint)
            # throttle rate
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.02))

    arr_obs = np.vstack(obs_list) if obs_list else np.zeros((0,10), dtype=np.float32)
    arr_tms = np.array(t_ms_list, dtype=np.int32)
    arr_cp  = np.array(cp_list, dtype=np.int32)

    np.savez_compressed(args.out, obs=arr_obs, t_ms=arr_tms, cp=arr_cp, meta=dict(hz=args.hz))
    print(f"Saved {arr_obs.shape[0]} frames to {args.out}")


if __name__ == '__main__':
    main()
