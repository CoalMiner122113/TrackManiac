#!/usr/bin/env python3
"""
Python client for the TMInterface RLBridge plugin (RLBridge.as).

- Connects to 127.0.0.1:<port> (default 5555)
- Receives fixed-size telemetry frames each physics tick
- Sends action packets (steer, throttle, brake, air_pitch, air_roll)
- Convenience test utilities (ping, constant drive, random drive, respawn/giveup)

Match this with the AngelScript plugin you installed in:
  Documents/TMInterface/Plugins/RLBridge.as

Telemetry frame layout (little-endian, 60 bytes):
  uint32 magic = 0x544D4652 ('TMFR')
  uint32 seq
  int32  race_time_ms
  float  pos_x, pos_y, pos_z
  float  yaw, pitch, roll
  float  vel_x, vel_y, vel_z
  float  speed_kmh
  int32  cur_checkpoint
  uint8  on_ground
  uint8  just_crossed_cp
  uint8  race_finished
  uint8  lateral_contact

Action protocol (to server):
  0x01 + 5 * float  -> set (steer[-1..1], throttle[0..1], brake[0..1], air_pitch[-1..1], air_roll[-1..1])
  0x02              -> Respawn()
  0x03              -> GiveUp()
  0x04              -> Horn()
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

MAGIC = 0x544D4652  # 'TMFR'
_FRAME_SIZE = 60
_MAGIC_BYTES = struct.pack('<I', MAGIC)

# struct format (little-endian)
# IIi fff fff fff f i 4B
_FRAME_FMT = '<IIi fff fff fff f i 4B'

@dataclass
class Telemetry:
    seq: int
    race_time_ms: int
    pos: Tuple[float, float, float]
    yaw: float
    pitch: float
    roll: float
    vel: Tuple[float, float, float]
    speed_kmh: float
    checkpoint: int
    on_ground: bool
    just_crossed_cp: bool
    race_finished: bool
    lateral_contact: bool

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

    @staticmethod
    def parse(buf: bytes) -> 'Telemetry':
        if len(buf) != _FRAME_SIZE:
            raise ValueError(f"Telemetry frame must be {_FRAME_SIZE} bytes, got {len(buf)}")
        # First 4 bytes are magic; already validated by client before calling parse
        _, seq, race_time_ms, px, py, pz, yaw, pitch, roll, vx, vy, vz, speed_kmh, cp, on_g, just_cp, finished, lateral = struct.unpack(
            _FRAME_FMT, buf
        )
        return Telemetry(
            seq=seq,
            race_time_ms=race_time_ms,
            pos=(px, py, pz),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            vel=(vx, vy, vz),
            speed_kmh=speed_kmh,
            checkpoint=cp,
            on_ground=bool(on_g),
            just_crossed_cp=bool(just_cp),
            race_finished=bool(finished),
            lateral_contact=bool(lateral),
        )


class RLBridgeClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 5555, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._recv_buf = bytearray()
        self._lock = threading.Lock()

    # ---------------- Connection -----------------
    def connect(self) -> None:
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        # Disable Nagle for lower latency
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.host, self.port))
        self.sock = s

    def close(self) -> None:
        with self._lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = None
            self._recv_buf.clear()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------- Telemetry receive -----------------
    def _recv_into_buffer(self) -> None:
        assert self.sock is not None
        try:
            chunk = self.sock.recv(4096)
        except socket.timeout:
            return
        if not chunk:
            raise ConnectionError("Socket closed by server")
        self._recv_buf.extend(chunk)

    def recv_frame(self, block: bool = True) -> Optional[Telemetry]:
        """Receive one telemetry frame. If block=False, returns None when no full frame available."""
        if self.sock is None:
            raise RuntimeError("Not connected")

        deadline = None if block else time.time()
        while True:
            # Try to parse a frame from buffer
            frame = self._try_parse_one()
            if frame is not None:
                return frame

            # Need more data
            if not block:
                return None

            # Respect timeout if set on socket
            self._recv_into_buffer()

            # Optional safety to avoid infinite loops if block=True and timeout was None
            if deadline is not None and time.time() > deadline:
                return None

    def _try_parse_one(self) -> Optional[Telemetry]:
        # Find magic in buffer; resync if necessary
        buf = self._recv_buf
        idx = buf.find(_MAGIC_BYTES)
        if idx == -1:
            # No magic yet; keep buffer bounded to avoid growth
            if len(buf) > 65536:
                del buf[:len(buf) - 4]
            return None
        if idx > 0:
            # Drop leading noise
            del buf[:idx]
        if len(buf) < _FRAME_SIZE:
            return None
        frame_bytes = bytes(buf[:_FRAME_SIZE])
        # Validate magic
        if frame_bytes[:4] != _MAGIC_BYTES:
            # Shouldn't happen because we aligned, but double-check
            del buf[:4]
            return None
        # Pop frame from buffer
        del buf[:_FRAME_SIZE]
        return Telemetry.parse(frame_bytes)

    # ---------------- Actions -----------------
    def send_action(self, steer: float, throttle: float, brake: float, air_pitch: float = 0.0, air_roll: float = 0.0) -> None:
        if self.sock is None:
            raise RuntimeError("Not connected")
        steer = max(-1.0, min(1.0, float(steer)))
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        air_pitch = max(-1.0, min(1.0, float(air_pitch)))
        air_roll = max(-1.0, min(1.0, float(air_roll)))
        pkt = b"\x01" + struct.pack('<fffff', steer, throttle, brake, air_pitch, air_roll)
        self.sock.sendall(pkt)

    def respawn(self) -> None:
        if self.sock is None:
            raise RuntimeError("Not connected")
        self.sock.sendall(b"\x02")

    def give_up(self) -> None:
        if self.sock is None:
            raise RuntimeError("Not connected")
        self.sock.sendall(b"\x03")

    def horn(self) -> None:
        if self.sock is None:
            raise RuntimeError("Not connected")
        self.sock.sendall(b"\x04")

    # ---------------- Utility: Reader thread -----------------
    def start_reader(self, callback: Callable[[Telemetry], None]) -> threading.Event:
        """Start a background reader that invokes callback for each frame. Returns a stop_event."""
        stop_event = threading.Event()

        def _run():
            try:
                while not stop_event.is_set():
                    frame = self.recv_frame(block=True)
                    if frame is not None:
                        callback(frame)
            except Exception as e:
                # Propagate error via callback with None? For simplicity, print.
                print(f"[reader] error: {e}")

        t = threading.Thread(target=_run, name="RLBridgeReader", daemon=True)
        t.start()
        return stop_event

    # ---------------- Tests -----------------
    def test_ping(self, seconds: float = 2.0) -> None:
        """Read frames for N seconds and report rate + sample frame."""
        t0 = time.time()
        n = 0
        sample: Optional[Telemetry] = None
        while time.time() - t0 < seconds:
            fr = self.recv_frame(block=True)
            if fr is None:
                continue
            n += 1
            if sample is None:
                sample = fr
        dt = max(1e-6, time.time() - t0)
        hz = n / dt
        print(f"Ping: received {n} frames in {dt:.2f}s ({hz:.1f} Hz)")
        if sample:
            print("Sample frame:")
            print(
                f"  seq={sample.seq} t={sample.race_time_ms}ms cp={sample.checkpoint} finish={sample.race_finished}\n"
                f"  pos=({sample.pos[0]:.2f},{sample.pos[1]:.2f},{sample.pos[2]:.2f})"
                f"  yaw/pitch/roll=({sample.yaw:.3f},{sample.pitch:.3f},{sample.roll:.3f})\n"
                f"  vel=({sample.vel[0]:.2f},{sample.vel[1]:.2f},{sample.vel[2]:.2f}) speed={sample.speed_kmh:.1f} km/h"
            )

    def test_constant_drive(self, duration: float = 3.0, steer: float = 0.0, throttle: float = 1.0, brake: float = 0.0) -> None:
        """Send a constant action for duration seconds, while consuming frames."""
        t0 = time.time()
        sent = 0
        while time.time() - t0 < duration:
            self.send_action(steer, throttle, brake)
            sent += 1
            # Consume at least one frame to keep buffers healthy
            _ = self.recv_frame(block=False)
            time.sleep(1.0 / 60.0)
        print(f"Constant drive: sent {sent} action packets in {duration:.2f}s")

    def test_random_drive(self, duration: float = 5.0) -> None:
        """Jitter steering randomly, full throttle, for duration seconds."""
        t0 = time.time()
        while time.time() - t0 < duration:
            steer = random.uniform(-1, 1)
            throttle = 1.0
            brake = 0.0
            self.send_action(steer, throttle, brake)
            _ = self.recv_frame(block=False)
            time.sleep(1.0 / 30.0)
        print("Random drive complete")

    def test_respawn(self) -> None:
        print("Respawn test: sending respawn command")
        self.respawn()

    def test_giveup(self) -> None:
        print("GiveUp test: sending give-up (restart) command")
        self.give_up()


# ---------------- CLI -----------------

def _main():
    parser = argparse.ArgumentParser(description="TMNF RLBridge Python client")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--timeout', type=float, default=1.0)

    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('ping', help='Read frames for a couple seconds and report fps')

    p_const = sub.add_parser('const', help='Drive constant action for N seconds')
    p_const.add_argument('--duration', type=float, default=3.0)
    p_const.add_argument('--steer', type=float, default=0.0)
    p_const.add_argument('--throttle', type=float, default=1.0)
    p_const.add_argument('--brake', type=float, default=0.0)

    p_rand = sub.add_parser('random', help='Random steering, throttle 1.0')
    p_rand.add_argument('--duration', type=float, default=5.0)

    sub.add_parser('respawn', help='Send Respawn()')
    sub.add_parser('giveup', help='Send GiveUp()')
    sub.add_parser('horn', help='Beep beep')

    args = parser.parse_args()

    with RLBridgeClient(args.host, args.port, args.timeout) as cli:
        if args.cmd == 'ping':
            cli.test_ping(2.0)
        elif args.cmd == 'const':
            cli.test_constant_drive(args.duration, args.steer, args.throttle, args.brake)
        elif args.cmd == 'random':
            cli.test_random_drive(args.duration)
        elif args.cmd == 'respawn':
            cli.test_respawn()
        elif args.cmd == 'giveup':
            cli.test_giveup()
        elif args.cmd == 'horn':
            cli.horn()
        else:
            raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == '__main__':
    _main()
