# tws_simulator.py
"""
Lightweight Python implementation of a TWSSimulator-like interface.

This is intentionally pure-Python and aimed at local development / testing.
It implements the methods and attributes expected by the GUI:
- TWSSimulator(sample_rate=44100)
- generate_wav(filename, freq_left=440, freq_right=880, duration=2.0)   # tone generator
- generate_anc_wav(filename, signal_freq=440, noise_freq=200, ...)
- process_audio() -> (left, right)
- save_output(left, right, out_l_path, out_r_path)
- get_battery_levels() -> (master_pct, slave_pct)

It also exposes the following writable attributes for the GUI:
- packet_loss_prob    # 0.0..1.0
- jitter_ms           # integer/float
- anc_enabled         # bool
- sample_rate         # int
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import threading
import time
import warnings

# optional dependency for writing WAVs
try:
    import scipy.io.wavfile as wavfile  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

__all__ = ["TWSSimulator"]

@dataclass
class TWSSimulator:
    """
    Simple simulator class suitable for the GUI.

    Designed to be a direct drop-in replacement for development/testing.
    Not intended for production-quality signal processing.
    """
    sample_rate: int = 44100

    # runtime-configurable attributes (GUI will set these)
    packet_loss_prob: float = 0.0   # 0.0 .. 1.0
    jitter_ms: float = 0.0          # jitter magnitude in milliseconds
    anc_enabled: bool = False

    # internal storage
    _left: Optional[np.ndarray] = None
    _right: Optional[np.ndarray] = None
    _battery_master: float = 100.0
    _battery_slave: float = 100.0

    def __post_init__(self):
        # start a background thread to simulate battery drain (low-cost thread)
        self._battery_lock = threading.Lock()
        self._battery_stop = threading.Event()
        self._battery_thread = threading.Thread(target=self._battery_worker, daemon=True)
        self._battery_thread.start()

    # -------------------------
    # Generation helpers
    # -------------------------
    def generate_wav(self, filename: str, freq_left: float = 440.0, freq_right: float = 880.0,
                     duration: float = 2.0) -> None:
        """
        Generate a simple stereo tone and store it internally.
        filename: path where the sim *may* save an input file (ignored but accepted).
        freq_left, freq_right: frequencies for each channel.
        duration: seconds
        """
        if duration <= 0:
            raise ValueError("duration must be > 0")
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        left = 0.6 * np.sin(2.0 * np.pi * float(freq_left) * t)
        right = 0.6 * np.sin(2.0 * np.pi * float(freq_right) * t)
        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

    def generate_anc_wav(self, filename: str, signal_freq: float = 440.0, noise_freq: float = 200.0,
                         signal_amp: float = 0.8, noise_amp: float = 0.3, duration: float = 2.0) -> None:
        """
        Generate a simple ANC test waveform: left=signal+noise, right=noise.
        """
        if duration <= 0:
            raise ValueError("duration must be > 0")
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        signal = signal_amp * np.sin(2.0 * np.pi * signal_freq * t)
        noise = noise_amp * np.sin(2.0 * np.pi * noise_freq * t)
        self._left = (signal + noise).astype(np.float32)
        self._right = noise.astype(np.float32)

    def load_input(self, filename: str) -> None:
        """
        Optionally load an input WAV (not required). If scipy is not available, this is a no-op.
        """
        if not HAVE_SCIPY:
            warnings.warn("scipy not available: load_input() is a no-op")
            return
        sr, data = wavfile.read(filename)
        if sr != self.sample_rate:
            warnings.warn("sample rate mismatch: file sr=%d sim sr=%d; data will be resampled naively" % (sr, self.sample_rate))
            # naive resampling: nearest neighbor length scaling (not high-quality)
            factor = int(round(self.sample_rate / sr))
            data = np.repeat(data, factor, axis=0)
        # normalize and split to float32 stereo
        data = data.astype(np.float32)
        if data.ndim == 1:
            left = right = data / np.max(np.abs(data) + 1e-12)
        else:
            left = data[:, 0] / (np.max(np.abs(data[:, 0])) + 1e-12)
            right = data[:, 1] / (np.max(np.abs(data[:, 1])) + 1e-12)
        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

    # -------------------------
    # Processing & effects
    # -------------------------
    def process_audio(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply configured packet loss and jitter to the internally stored audio and return (left, right).
        If no internal audio present, returns two empty arrays.
        """
        if self._left is None or self._right is None:
            return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

        L = self._left.copy()
        R = self._right.copy()

        # packet loss: randomly zero samples (simple sample-wise drop to emulate packet loss)
        p = float(self.packet_loss_prob)
        if p > 0.0:
            if p < 0.0 or p > 1.0:
                raise ValueError("packet_loss_prob must be in [0.0, 1.0]")
            mask = np.random.rand(L.size) >= p
            L = L * mask
            R = R * mask

        # jitter: block-wise shifting of samples by a small random offset
        jms = float(self.jitter_ms)
        if jms > 0.0:
            max_shift = int(round((jms / 1000.0) * self.sample_rate))
            if max_shift > 0:
                block = max(1, int(self.sample_rate * 0.02))  # 20ms blocks
                outL = np.zeros_like(L)
                outR = np.zeros_like(R)
                i = 0
                while i < L.size:
                    j = min(i + block, L.size)
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    dst0 = i + shift
                    dst1 = dst0 + (j - i)
                    # compute overlap with [0, L.size)
                    dst_start = max(dst0, 0)
                    src_start = max(0, -dst0)
                    copy_len = min(L.size - dst_start, (j - i) - src_start)
                    if copy_len > 0 and dst_start < L.size:
                        outL[dst_start:dst_start + copy_len] = L[i + src_start : i + src_start + copy_len]
                        outR[dst_start:dst_start + copy_len] = R[i + src_start : i + src_start + copy_len]
                    i = j
                L = outL
                R = outR

        # simple ANC behavior (if enabled, subtract right from left small scale)
        if self.anc_enabled:
            # naive cancellation: attempt to subtract right from left scaled
            # this is purely illustrative and not real ANC
            L = L - 0.3 * R
            L = np.clip(L, -1.0, 1.0)

        return (L.astype(np.float32), R.astype(np.float32))

    # -------------------------
    # IO
    # -------------------------
    def save_output(self, left: np.ndarray, right: np.ndarray, out_l: str, out_r: str) -> None:
        """
        Save left & right arrays to WAV files. Uses scipy if available; else raises.
        left/right expected float arrays in [-1,1].
        """
        if not HAVE_SCIPY:
            raise RuntimeError("scipy not available: cannot save WAV files from simulator")
        left_i16 = (np.clip(left, -1.0, 1.0) * 32767).astype(np.int16)
        right_i16 = (np.clip(right, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(out_l, self.sample_rate, left_i16)
        wavfile.write(out_r, self.sample_rate, right_i16)

    # -------------------------
    # Playback convenience (optional)
    # -------------------------
    def play_audio(self, left: np.ndarray, right: np.ndarray) -> None:
        """
        Convenience wrapper using sounddevice if present. Non-blocking behaviour not guaranteed.
        """
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            raise RuntimeError("sounddevice not available for play_audio()")
        max_len = max(len(left), len(right))
        left_p = np.pad(left, (0, max_len - len(left)))
        right_p = np.pad(right, (0, max_len - len(right)))
        stereo = np.stack((left_p, right_p), axis=1)
        sd.play(stereo, samplerate=self.sample_rate)
        sd.wait()

    # -------------------------
    # Battery simulation
    # -------------------------
    def _battery_worker(self) -> None:
        """
        Small background thread to slowly decrease battery levels while the process runs.
        This is harmless and lightweight.
        """
        while not self._battery_stop.is_set():
            time.sleep(5.0)
            with self._battery_lock:
                # degrade slowly; clip [0, 100]
                self._battery_master = max(0.0, self._battery_master - 0.05)
                self._battery_slave = max(0.0, self._battery_slave - 0.03)

    def get_battery_levels(self) -> Tuple[float, float]:
        with self._battery_lock:
            return (float(self._battery_master), float(self._battery_slave))

    def shutdown(self) -> None:
        """
        Stop background threads cleanly (call before process exit if desired).
        """
        self._battery_stop.set()
        if hasattr(self, "_battery_thread") and self._battery_thread.is_alive():
            self._battery_thread.join(timeout=1.0)
