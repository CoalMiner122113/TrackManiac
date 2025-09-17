#!/usr/bin/env python3
"""
Gymnasium environment for TrackMania Nations Forever via TMInterface RLBridge.

Depends on the companion client file: RLBridgeClient.py

- Observations: rich state vector (speed, yaw_sin/cos, pitch, roll, v_local_fwd/lat, on_ground, cp_idx, t_scaled)
- Actions: [steer(-1..1), throttle(0..1), brake(0..1), air_pitch(-1..1), air_roll(-1..1)]
- Rewards: pace shaping + checkpoint bonuses + split improvements vs. personal best
- Resets: GiveUp -> wait for race_time reset -> small stabilization window

Usage (quick test rollout):
    python tmnf_gym_env.py --host 127.0.0.1 --port 5555 --rollout 5

Minimal PPO (separate script or inline at bottom): see `if __name__ == "__main__":` section for example.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

# Import the existing client
from RLBridgeClient import RLBridgeClient, Telemetry


# ---------------- Split timer book -----------------
class SplitBook:
    """Tracks best time at each checkpoint and gives a reward for improvements."""
    def __init__(self):
        self.best: dict[int, float] = {}

    def reward(self, cp_idx: int, t_sec: float) -> float:
        # First time hitting this CP -> small bonus and set PB
        if cp_idx not in self.best:
            self.best[cp_idx] = t_sec
            return 0.5
        prev = self.best[cp_idx]
        delta = prev - t_sec  # positive if faster
        # Bounded improvement signal
        r = 0.3 * float(np.clip(delta, -1.0, 1.0))
        if t_sec < prev:
            self.best[cp_idx] = t_sec
        return r


# ---------------- Utility math -----------------
_DEF_MAX_SPEED_KMH = 300.0
_DEF_MAX_EPISODE_TIME = 90.0  # seconds


def _normalize_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 5555
    frame_skip: int = 3
    max_episode_time: float = _DEF_MAX_EPISODE_TIME
    pace_coef: float = 0.0025  # per m/s
    cp_bonus: float = 0.2
    idle_penalty_after_s: float = 3.0
    idle_penalty: float = 0.05
    crash_penalty: float = 0.2  # uses lateral_contact flag as proxy


class TMNFEnv(gym.Env):
    """Gymnasium-compatible env powered by RLBridgeClient.

    Observation (10, ): [speed, yaw_sin, yaw_cos, pitch, roll, v_fwd, v_lat, on_ground, cp_scaled, t_scaled]
    Action (5, ): [steer, throttle, brake, air_pitch, air_roll]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.client = RLBridgeClient(cfg.host, cfg.port, timeout=1.0)

        # Spaces
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0, -1.0, -1.0], np.float32),
                                           high=np.array([+1.0, 1.0, 1.0, +1.0, +1.0], np.float32),
                                           dtype=np.float32)

        # Episode state
        self.split_book = SplitBook()
        self._last_cp = -1
        self._idle_timer = 0.0
        self._t_start_monotonic = None

    # --------------- Gym API -----------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Ensure connection
        self.client.connect()

        # Hard reset the run
        self.client.give_up()
        # Wait for race_time to reset near zero and frames to flow
        t0 = time.time()
        first: Optional[Telemetry] = None
        while True:
            fr = self.client.recv_frame(block=True)
            if fr is not None:
                first = fr
                if fr.race_time_ms <= 250:  # close to start
                    break
            if time.time() - t0 > 5.0:
                break
        # Stabilize a couple frames
        for _ in range(3):
            _ = self.client.recv_frame(block=True)

        self._last_cp = -1 if first is None else first.checkpoint
        self._idle_timer = 0.0
        self._t_start_monotonic = time.time()

        obs = self._obs_from(fr)
        info = {"cp": fr.checkpoint if fr else -1}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Frame-skip for smoother control
        fr: Optional[Telemetry] = None
        for _ in range(self.cfg.frame_skip):
            self.client.send_action(*action.tolist())
            fr = self.client.recv_frame(block=True)

        assert fr is not None, "No telemetry frame received"

        obs = self._obs_from(fr)
        r, cp_hit = self._compute_reward(fr)

        # Terminations
        t_game = fr.race_time_ms / 1000.0
        terminated = False  # You may switch to finished flag if you want finish to be terminal
        truncated = t_game > self.cfg.max_episode_time

        info = {
            "cp": fr.checkpoint,
            "split": cp_hit,
            "speed_kmh": fr.speed_kmh,
            "in_race": True,
        }
        return obs, r, terminated, truncated, info

    # --------------- Internals -----------------
    def _obs_from(self, fr: Telemetry) -> np.ndarray:
        # Yaw/pitch/roll
        yaw = _normalize_angle(fr.yaw)
        pitch = float(np.clip(fr.pitch / 0.6, -1.0, 1.0))
        roll = float(np.clip(fr.roll / 0.8, -1.0, 1.0))

        # Rotate world velocity into car-local using yaw only (stable & cheap)
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = fr.vel
        v_fwd = c * vx + s * vz
        v_lat = -s * vx + c * vz
        v_fwd_n = float(np.tanh(v_fwd / 50.0))
        v_lat_n = float(np.tanh(v_lat / 50.0))

        speed_n = float(np.clip(fr.speed_kmh / _DEF_MAX_SPEED_KMH, 0.0, 1.0))
        yaw_s, yaw_c = math.sin(yaw), math.cos(yaw)
        on_ground = 1.0 if fr.on_ground else 0.0
        cp_scaled = float(np.tanh(fr.checkpoint / 10.0))
        t_scaled = float(np.clip((fr.race_time_ms / 1000.0) / self.cfg.max_episode_time, 0.0, 1.0))

        return np.array([speed_n, yaw_s, yaw_c, pitch, roll, v_fwd_n, v_lat_n,
                         on_ground, cp_scaled, t_scaled], dtype=np.float32)

    def _compute_reward(self, fr: Telemetry) -> Tuple[float, Optional[int]]:
        r = 0.0
        # Pace shaping (encourage moving forward fast)
        r += self.cfg.pace_coef * max(0.0, fr.speed_kmh / 3.6)  # m/s

        cp_hit: Optional[int] = None
        if fr.checkpoint > self._last_cp:
            # Crossed a new CP
            t_sec = fr.race_time_ms / 1000.0
            r += self.cfg.cp_bonus
            r += self.split_book.reward(fr.checkpoint, t_sec)
            self._last_cp = fr.checkpoint
            self._idle_timer = 0.0
            cp_hit = fr.checkpoint
        else:
            self._idle_timer += self.cfg.frame_skip / 60.0

        # Simple crash proxy
        if fr.lateral_contact:
            r -= self.cfg.crash_penalty

        if self._idle_timer > self.cfg.idle_penalty_after_s:
            r -= self.cfg.idle_penalty

        return float(np.clip(r, -1.0, 1.0)), cp_hit

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# ---------------- Quick test / demo -----------------

def _rollout(env: TMNFEnv, seconds: float = 5.0):
    import random
    t0 = time.time()
    obs, info = env.reset()
    print("reset info:", info)
    steps = 0
    while time.time() - t0 < seconds:
        # simple heuristic: steer small random, throttle on
        a = np.array([random.uniform(-0.2, 0.2), 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if steps % 20 == 0:
            print(f"step={steps} r={r:.3f} cp={info['cp']} split={info['split']} speed={info['speed_kmh']:.1f}")
        if term or trunc:
            print("episode end: term=", term, " trunc=", trunc)
            obs, info = env.reset()
            steps = 0
    env.close()


def _ppo_example(env: TMNFEnv):
    """Minimal PPO example with Stable-Baselines3."""
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        print("Stable-Baselines3 not installed:", e)
        return
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=8192,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.003,
        clip_range=0.2,
        verbose=1,
    )
    model.learn(total_timesteps=100_000)
    model.save("tmnf_ppo_demo.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--rollout", type=float, default=5.0, help="seconds to run a random rollout; 0 to skip")
    parser.add_argument("--train_demo", action="store_true", help="run a short PPO demo (100k steps)")
    args = parser.parse_args()

    env = TMNFEnv(EnvConfig(host=args.host, port=args.port))
    if args.rollout > 0:
        _rollout(env, seconds=args.rollout)
    if args.train_demo:
        _ppo_example(env)
