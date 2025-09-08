"""
tws_simulator.py (updated)

A TWSSimulator-like interface that prefers C++ modules when available.
Keeps the same public API as the original pure-Python version.
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

# Try to import C++ modules; fall back gracefully
_HAVE_CPP = True
try:
    import audio_processor
    import battery_model
    import wav_generator
except Exception as e:
    _HAVE_CPP = False
    _cpp_err = e
    warnings.warn(f"C++ modules not loaded; using pure-Python simulator. Reason: {e}")

__all__ = ["TWSSimulator"]

@dataclass
class TWSSimulator:
    sample_rate: int = 44100

    packet_loss_prob: float = 0.0
    jitter_ms: float = 0.0
    anc_enabled: bool = False

    _left: Optional[np.ndarray] = None
    _right: Optional[np.ndarray] = None

    # internal battery objects (C++ BatteryModel or fallback numeric)
    _battery_master_obj: Optional[object] = None
    _battery_slave_obj: Optional[object] = None

    def __post_init__(self):
        # instantiate C++ objects if available
        if _HAVE_CPP:
            try:
                # AudioProcessor(filter_len, mu, seed)
                self._ap = audio_processor.AudioProcessor(32, 0.1, 42)
            except Exception:
                self._ap = None
                warnings.warn("Failed to create audio_processor.AudioProcessor; will use Python processing.")

            try:
                self._battery_master_obj = battery_model.BatteryModel(100.0, 100.0)
                self._battery_slave_obj = battery_model.BatteryModel(100.0, 100.0)
            except Exception:
                self._battery_master_obj = None
                self._battery_slave_obj = None
                warnings.warn("Failed to instantiate battery_model.BatteryModel; battery features limited.")

            try:
                # wav_generator.WavGenerator(sample_rate, duration)
                # keep a generator with default duration for convenience
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, 2.0)
            except Exception:
                self._wavgen = None
                warnings.warn("Failed to instantiate wav_generator.WavGenerator.")
        else:
            self._ap = None
            self._battery_master_obj = None
            self._battery_slave_obj = None
            self._wavgen = None

        # battery thread for UI realism
        self._battery_lock = threading.Lock()
        self._battery_stop = threading.Event()
        self._battery_thread = threading.Thread(target=self._battery_worker, daemon=True)
        self._battery_thread.start()

    # -------------------------
    # Generation helpers
    # -------------------------
    def generate_wav(self, filename: str, freq_left: float = 440.0, freq_right: float = 880.0,
                     duration: float = 2.0) -> None:
        """Generate a simple stereo tone and store internally. If C++ wav_generator is available, use it."""
        dur = float(duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")

        if _HAVE_CPP and self._wavgen is not None:
            try:
                # re-create generator with desired duration
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, dur)
                self._wavgen.generate_stereo_wav(filename, float(freq_left), float(freq_right), 0.6)
                wavdata = self._wavgen.read_wav(filename)
                self._left, self._right = self._wavdata_to_arrays(wavdata)
                return
            except Exception as e:
                warnings.warn(f"C++ wav generation failed; falling back to numpy: {e}")

        # fallback
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        left = 0.6 * np.sin(2.0 * np.pi * float(freq_left) * t)
        right = 0.6 * np.sin(2.0 * np.pi * float(freq_right) * t)
        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

        # try to save to filename using scipy if available
        if filename and HAVE_SCIPY:
            left_i16 = (np.clip(self._left, -1.0, 1.0) * 32767).astype(np.int16)
            right_i16 = (np.clip(self._right, -1.0, 1.0) * 32767).astype(np.int16)
            stereo = np.stack((left_i16, right_i16), axis=1)
            wavfile.write(filename, self.sample_rate, stereo)

    def generate_anc_wav(self, filename: str, signal_freq: float = 440.0, noise_freq: float = 200.0,
                         signal_amp: float = 0.8, noise_amp: float = 0.3, duration: float = 2.0) -> None:
        """Generate ANC test waveform: left=signal+noise, right=noise."""
        dur = float(duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")

        if _HAVE_CPP and self._wavgen is not None:
            try:
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, dur)
                self._wavgen.generate_anc_wav(filename, float(signal_freq), float(noise_freq),
                                              float(signal_amp), float(noise_amp))
                wavdata = self._wavgen.read_wav(filename)
                self._left, self._right = self._wavdata_to_arrays(wavdata)
                return
            except Exception as e:
                warnings.warn(f"C++ ANC WAV generation failed; falling back to numpy: {e}")

        # fallback generation
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        signal = float(signal_amp) * np.sin(2.0 * np.pi * float(signal_freq) * t)
        noise = float(noise_amp) * np.sin(2.0 * np.pi * float(noise_freq) * t)
        self._left = (signal + noise).astype(np.float32)
        self._right = noise.astype(np.float32)

    def load_input(self, filename: str) -> None:
        """Load input WAV. If C++ wav_generator.read_wav available prefer it, else use scipy."""
        if _HAVE_CPP and self._wavgen is not None:
            try:
                wavdata = self._wavgen.read_wav(filename)
                self._left, self._right = self._wavdata_to_arrays(wavdata)
                return
            except Exception as e:
                warnings.warn(f"C++ read_wav failed; falling back to scipy: {e}")

        if not HAVE_SCIPY:
            warnings.warn("scipy not available: load_input() is a no-op")
            return

        sr, data = wavfile.read(filename)
        if sr != self.sample_rate:
            warnings.warn(f"sample rate mismatch: file sr={sr} sim sr={self.sample_rate}; naive resample will be applied")
            factor = int(round(self.sample_rate / sr))
            data = np.repeat(data, factor, axis=0)

        data = data.astype(np.float32)
        if data.ndim == 1:
            left = right = data / (np.max(np.abs(data)) + 1e-12)
        else:
            left = data[:, 0] / (np.max(np.abs(data[:, 0])) + 1e-12)
            right = data[:, 1] / (np.max(np.abs(data[:, 1])) + 1e-12)
        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

    def _wavdata_to_arrays(self, wavdata) -> Tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(wavdata.samples, dtype=np.float32)
        channels = int(wavdata.channels)
        if channels == 1:
            return samples, samples
        elif channels == 2:
            samples = samples.reshape((-1, 2))
            return samples[:, 0].astype(np.float32), samples[:, 1].astype(np.float32)
        else:
            raise RuntimeError("Unsupported channel count in WAV")

    # -------------------------
    # Processing & effects
    # -------------------------
    def process_audio(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply packet loss, jitter and optional ANC using C++ audio_processor if available."""
        if self._left is None or self._right is None:
            return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

        L = self._left.copy()
        R = self._right.copy()

        # Ensure float32 numpy arrays
        L = np.asarray(L, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)

        # Try C++ path first
        if _HAVE_CPP and self._ap is not None:
            try:
                if self.packet_loss_prob > 0.0:
                    L = np.asarray(self._ap.apply_packet_loss(L, float(self.packet_loss_prob), int(self.sample_rate), 20), dtype=np.float32)
                    R = np.asarray(self._ap.apply_packet_loss(R, float(self.packet_loss_prob), int(self.sample_rate), 20), dtype=np.float32)

                if self.jitter_ms > 0.0:
                    L = np.asarray(self._ap.apply_jitter(L, float(self.jitter_ms), int(self.sample_rate), 20), dtype=np.float32)
                    R = np.asarray(self._ap.apply_jitter(R, float(self.jitter_ms), int(self.sample_rate), 20), dtype=np.float32)

                if self.anc_enabled:
                    # apply_anc returns processed left (error) as numpy array
                    L = np.asarray(self._ap.apply_anc(L, R), dtype=np.float32)
                    L = np.clip(L, -1.0, 1.0)

                return (L.astype(np.float32), R.astype(np.float32))
            except Exception as e:
                warnings.warn(f"C++ processing failed; falling back to Python path: {e}")

        # Pure-Python fallback (same as your previous implementation)
        p = float(self.packet_loss_prob)
        if p > 0.0:
            if p < 0.0 or p > 1.0:
                raise ValueError("packet_loss_prob must be in [0.0, 1.0]")
            mask = np.random.rand(L.size) >= p
            L = L * mask
            R = R * mask

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
                    dst_start = max(dst0, 0)
                    src_start = max(0, -dst0)
                    copy_len = min(L.size - dst_start, (j - i) - src_start)
                    if copy_len > 0 and dst_start < L.size:
                        outL[dst_start:dst_start + copy_len] = L[i + src_start : i + src_start + copy_len]
                        outR[dst_start:dst_start + copy_len] = R[i + src_start : i + src_start + copy_len]
                    i = j
                L = outL
                R = outR

        if self.anc_enabled:
            L = L - 0.3 * R
            L = np.clip(L, -1.0, 1.0)

        return (L.astype(np.float32), R.astype(np.float32))

    # -------------------------
    # IO
    # -------------------------
    def save_output(self, left: np.ndarray, right: np.ndarray, out_l: str, out_r: str) -> None:
        if not HAVE_SCIPY:
            raise RuntimeError("scipy not available: cannot save WAV files from simulator")
        left_i16 = (np.clip(left, -1.0, 1.0) * 32767).astype(np.int16)
        right_i16 = (np.clip(right, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(out_l, self.sample_rate, left_i16)
        wavfile.write(out_r, self.sample_rate, right_i16)

    def play_audio(self, left: np.ndarray, right: np.ndarray) -> None:
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
        while not self._battery_stop.is_set():
            time.sleep(5.0)
            with self._battery_lock:
                if _HAVE_CPP and self._battery_master_obj is not None and self._battery_slave_obj is not None:
                    try:
                        # small time step
                        self._battery_master_obj.update(5.0, bool(self.anc_enabled))
                        self._battery_slave_obj.update(5.0, bool(self.anc_enabled))
                        continue
                    except Exception:
                        # fall through to python based degradation
                        pass

                # python fallback counters (create if missing)
                if not hasattr(self, "_py_batt_master"):
                    self._py_batt_master = 100.0
                    self._py_batt_slave = 100.0
                self._py_batt_master = max(0.0, self._py_batt_master - 0.05)
                self._py_batt_slave = max(0.0, self._py_batt_slave - 0.03)

    def get_battery_levels(self) -> Tuple[float, float]:
        with self._battery_lock:
            if _HAVE_CPP and self._battery_master_obj is not None and self._battery_slave_obj is not None:
                try:
                    return (float(self._battery_master_obj.get_level()), float(self._battery_slave_obj.get_level()))
                except Exception:
                    pass
            if hasattr(self, "_py_batt_master"):
                return (float(self._py_batt_master), float(self._py_batt_slave))
            return (100.0, 100.0)

    def shutdown(self) -> None:
        self._battery_stop.set()
        if hasattr(self, "_battery_thread") and self._battery_thread.is_alive():
            self._battery_thread.join(timeout=1.0)
