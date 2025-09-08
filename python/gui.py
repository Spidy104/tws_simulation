"""
gui.py

Helper layer used by the GUI. Wraps the C++ pybind11 modules (if available)
and exposes a clean, numpy-friendly interface.

Exposed class: GUIModel
- Uses audio_processor.AudioProcessor, battery_model.BatteryModel, wav_generator.WavGenerator
- Gracefully falls back to Python implementations when C++ modules are not importable
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warnings
import sys
import os
import tempfile

# Always add the project "python" directory (next to this file) to sys.path
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)

# Add repo/src so compiled .so modules placed there can be imported as top-level modules
repo_root = os.path.dirname(this_dir)
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try to import C++ extension modules; set flags for fallback behavior
_HAVE_CPP = True
try:
    # Import names as top-level modules (the compiled .so files should be in src/)
    import audio_processor    # pybind11 module (src/audio_processor*.so)
    import battery_model      # pybind11 module (src/battery_model*.so)
    import wav_generator      # pybind11 module (src/wav_generator*.so)
    # If imports succeed we can use them; otherwise fallback
    print("Imported cpp mopdules")
except Exception as e:
    _HAVE_CPP = False
    _cpp_import_error = e
    warnings.warn(f"C++ modules not available, falling back to pure-Python behavior: {e}")


@dataclass
class GUIModel:
    """
    Wraps the C++ objects and exposes simple numpy-based functions used by the GUI.
    If C++ modules are missing it still provides the same API (but via Python code).
    """

    sample_rate: int = 44100
    duration: float = 2.0

    # runtime-configurable attributes (GUI will set these)
    packet_loss_prob: float = 0.0   # 0..1
    jitter_ms: float = 0.0
    anc_enabled: bool = False

    # internal objects
    _ap: Optional[object] = None
    _battery_master: Optional[object] = None
    _battery_slave: Optional[object] = None
    _wavgen: Optional[object] = None

    def __post_init__(self):
        # instantiate C++ objects when available; otherwise leave None and use Python fallbacks
        if _HAVE_CPP:
            try:
                # AudioProcessor(filter_len, mu, seed) - signature assumed from C++ binding
                self._ap = audio_processor.AudioProcessor(32, 0.1, 42)
            except Exception:
                self._ap = None
                warnings.warn("Failed to instantiate audio_processor.AudioProcessor; using Python fallbacks.")

            try:
                # BatteryModel(level_percent, capacity) - C++ wrapper provides this mapping
                self._battery_master = battery_model.BatteryModel(100.0, 100.0)
                self._battery_slave  = battery_model.BatteryModel(100.0, 100.0)
            except Exception:
                self._battery_master = None
                self._battery_slave  = None
                warnings.warn("Failed to instantiate battery_model.BatteryModel; battery fallback enabled.")

            try:
                # wav_generator.WavGenerator(sample_rate, duration)
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, float(self.duration))
            except Exception:
                self._wavgen = None
                warnings.warn("Failed to instantiate wav_generator.WavGenerator; wav generation will fall back to numpy.")
        else:
            # pure-Python fallback mode
            self._ap = None
            self._battery_master = None
            self._battery_slave  = None
            self._wavgen = None

        # last generated buffers
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None

        # clip inputs
        self.packet_loss_prob = float(np.clip(self.packet_loss_prob, 0.0, 1.0))
        self.jitter_ms = float(max(0.0, self.jitter_ms))

    # -------------------------
    # Wave generation helpers
    # -------------------------
    def _ensure_valid_filename(self, filename: str) -> str:
        """Return filename if non-empty else create a temporary name to use."""
        if filename and filename.strip():
            return filename
        # create a temporary file path and return it (do not delete immediately)
        tf = tempfile.NamedTemporaryFile(prefix="tws_gen_", suffix=".wav", delete=False)
        tf.close()
        return tf.name

    def generate_wav(self, filename: str, freq_left: float = 440.0, freq_right: float = 880.0,
                     duration: Optional[float] = None) -> None:
        """Generate a stereo tone and populate internal left/right arrays.
        If C++ wav generator is present it will try to write to disk (filename must be valid)."""
        dur = float(duration if duration is not None else self.duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")

        filename_safe = self._ensure_valid_filename(filename)

        # Try C++ generator first (if available)
        if _HAVE_CPP and self._wavgen is not None:
            try:
                # Recreate wavgen with new duration (bindings expect sample_rate, duration)
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, dur)
                self._wavgen.generate_stereo_wav(filename_safe, float(freq_left), float(freq_right), 0.8)
                wavdata = self._wavgen.read_wav(filename_safe)
                self._left, self._right = self._wavdata_to_arrays(wavdata)
                return
            except Exception as e:
                warnings.warn(f"C++ wav_generator failed, falling back to numpy generator: {e}")

        # Pure-Python fallback: simple sine tones
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        left = 0.6 * np.sin(2.0 * np.pi * float(freq_left) * t)
        right = 0.6 * np.sin(2.0 * np.pi * float(freq_right) * t)
        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

        # Save to file if requested
        if filename_safe:
            try:
                from scipy.io import wavfile
                stereo = np.stack((self._left, self._right), axis=1)
                wavfile.write(filename_safe, self.sample_rate, (stereo * 32767).astype(np.int16))
            except Exception:
                pass  # optional

    def generate_anc_wav(self, filename: str,
                         signal_freq: float = 440.0,
                         noise_freq: float = 200.0,
                         signal_amp: float = 0.8,
                         noise_amp: float = 0.3,
                         duration: Optional[float] = None) -> None:
        """Generate ANC test waveform: left = signal + noise, right = noise."""
        dur = float(duration if duration is not None else self.duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")

        filename_safe = self._ensure_valid_filename(filename)

        if _HAVE_CPP and self._wavgen is not None:
            try:
                self._wavgen = wav_generator.WavGenerator(self.sample_rate, dur)
                self._wavgen.generate_anc_wav(filename_safe,
                                              float(signal_freq),
                                              float(noise_freq),
                                              float(signal_amp),
                                              float(noise_amp))
                wavdata = self._wavgen.read_wav(filename_safe)
                self._left, self._right = self._wavdata_to_arrays(wavdata)
                return
            except Exception as e:
                warnings.warn(f"C++ generate_anc_wav failed, falling back to numpy generator: {e}")

        # Python fallback
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        signal = float(signal_amp) * np.sin(2.0 * np.pi * float(signal_freq) * t)
        noise  = float(noise_amp)  * np.sin(2.0 * np.pi * float(noise_freq) * t)
        self._left  = (signal + noise).astype(np.float32)
        self._right = noise.astype(np.float32)

    def generate_sweep(self, filename: str, start_freq: float = 200.0, end_freq: float = 2000.0,
                       duration: Optional[float] = None, method: str = "log") -> None:
        """Generate a sweep (left channel) and a slightly offset sweep for right channel.
        Methods: 'linear' or 'log' (logarithmic)."""
        dur = float(duration if duration is not None else self.duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")

        filename_safe = self._ensure_valid_filename(filename)

        # If C++ wav_generator supports sweeps in future, use it; otherwise numpy fallback:
        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        if method == "linear":
            freqs = np.linspace(start_freq, end_freq, t.size)
        else:
            freqs = np.logspace(np.log10(max(1.0, start_freq)), np.log10(max(1.0, end_freq)), t.size)

        left = 0.6 * np.sin(2.0 * np.pi * freqs * t)
        # give the right channel a slightly faster sweep for distinction
        freqs_r = freqs * 1.15
        right = 0.6 * np.sin(2.0 * np.pi * freqs_r * t)

        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

        # optionally save
        if filename_safe:
            try:
                from scipy.io import wavfile
                stereo = np.stack((self._left, self._right), axis=1)
                wavfile.write(filename_safe, self.sample_rate, (stereo * 32767).astype(np.int16))
            except Exception:
                pass

    def generate_preset(self, filename: str, preset: str = "white", duration: Optional[float] = None) -> None:
        """Generate a preset: 'white', 'pink' (approx), 'chord' or 'speech_like' (simple bursts)."""
        dur = float(duration if duration is not None else self.duration)
        if dur <= 0:
            raise ValueError("duration must be > 0")
        filename_safe = self._ensure_valid_filename(filename)

        t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
        if preset == "white":
            left = np.random.randn(t.size) * 0.2
            right = np.random.randn(t.size) * 0.2
        elif preset == "chord":
            left = 0.4 * np.sin(2*np.pi*440*t) + 0.25*np.sin(2*np.pi*550*t)
            right = 0.4 * np.sin(2*np.pi*660*t)
        elif preset == "speech_like":
            # simple bursts: bandlimited noise gated by envelope
            noise = np.random.randn(t.size) * 0.3
            env = np.abs(np.sin(np.linspace(0, 10*np.pi, t.size)))  # simple envelope
            left = noise * env
            right = noise * 0.6 * env
        else:
            # fallback to tone
            left = 0.6 * np.sin(2.0 * np.pi * 440.0 * t)
            right = 0.6 * np.sin(2.0 * np.pi * 880.0 * t)

        self._left = left.astype(np.float32)
        self._right = right.astype(np.float32)

        if filename_safe:
            try:
                from scipy.io import wavfile
                stereo = np.stack((self._left, self._right), axis=1)
                wavfile.write(filename_safe, self.sample_rate, (stereo * 32767).astype(np.int16))
            except Exception:
                pass

    def _wavdata_to_arrays(self, wavdata) -> Tuple[np.ndarray, np.ndarray]:
        """Convert WavData from wav_generator.read_wav to numpy left/right arrays.
        The pybind11 WavData has .samples (flat vector), .channels, .sample_rate."""
        samples = wavdata.samples
        channels = int(wavdata.channels)
        arr = np.asarray(samples, dtype=np.float32)
        if channels == 1:
            left = arr
            right = arr
        else:
            arr = arr.reshape((-1, channels))
            left = arr[:, 0]
            right = arr[:, 1]
        return left.astype(np.float32), right.astype(np.float32)

    # -------------------------
    # Processing pipeline
    # -------------------------
    def process_audio(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply packet loss, jitter, and (optional) ANC. Returns (left, right)."""
        if self._left is None or self._right is None:
            return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

        L = np.asarray(self._left, dtype=np.float32).copy()
        R = np.asarray(self._right, dtype=np.float32).copy()

        # Prefer C++ processing if available
        if _HAVE_CPP and self._ap is not None:
            try:
                if self.packet_loss_prob > 0.0:
                    L = np.asarray(self._ap.apply_packet_loss(L, float(self.packet_loss_prob), int(self.sample_rate), 20), dtype=np.float32)
                    R = np.asarray(self._ap.apply_packet_loss(R, float(self.packet_loss_prob), int(self.sample_rate), 20), dtype=np.float32)
                if self.jitter_ms > 0.0:
                    L = np.asarray(self._ap.apply_jitter(L, float(self.jitter_ms), int(self.sample_rate), 20), dtype=np.float32)
                    R = np.asarray(self._ap.apply_jitter(R, float(self.jitter_ms), int(self.sample_rate), 20), dtype=np.float32)
                if self.anc_enabled:
                    out = np.asarray(self._ap.apply_anc(L, R), dtype=np.float32)
                    L = np.clip(out, -1.0, 1.0)
                return (L.astype(np.float32), R.astype(np.float32))
            except Exception as e:
                warnings.warn(f"C++ audio processing failed, falling back to Python path: {e}")

        # Python fallback processing
        p = float(self.packet_loss_prob)
        if p > 0.0:
            mask = np.random.rand(L.size) >= p
            L = L * mask
            R = R * mask

        jms = float(self.jitter_ms)
        if jms > 0.0:
            max_shift = int(round((jms / 1000.0) * float(self.sample_rate)))
            if max_shift > 0:
                block = max(1, int(self.sample_rate * 0.02))
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
    # Saving helpers
    # -------------------------
    def save_output(self, left: np.ndarray, right: np.ndarray, out_l: str, out_r: str) -> None:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        try:
            from scipy.io import wavfile
            left_i16 = (np.clip(left, -1.0, 1.0) * 32767).astype(np.int16)
            right_i16 = (np.clip(right, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(out_l, self.sample_rate, left_i16)
            wavfile.write(out_r, self.sample_rate, right_i16)
        except Exception as e:
            raise RuntimeError("Unable to save WAV files: ensure scipy is installed") from e

    # -------------------------
    # Battery accessors
    # -------------------------
    def update_battery(self, duration_s: float) -> None:
        d = float(max(0.0, duration_s))
        if _HAVE_CPP and self._battery_master is not None and self._battery_slave is not None:
            try:
                self._battery_master.update(d, bool(self.anc_enabled))
                self._battery_slave.update(d, bool(self.anc_enabled))
                return
            except Exception:
                warnings.warn("C++ battery update failed; falling back to Python drain.")

        # fallback simple drain
        if not hasattr(self, "_py_batt_master"):
            self._py_batt_master = 100.0
            self._py_batt_slave  = 100.0
        self._py_batt_master = max(0.0, self._py_batt_master - (0.01 * d + (0.02 * d if self.anc_enabled else 0.0)))
        self._py_batt_slave  = max(0.0, self._py_batt_slave  - (0.01 * d))

    def get_battery_levels(self) -> Tuple[float, float]:
        if _HAVE_CPP and self._battery_master is not None and self._battery_slave is not None:
            try:
                return float(self._battery_master.get_level()), float(self._battery_slave.get_level())
            except Exception:
                warnings.warn("C++ get_battery_levels failed; using fallback values.")
        if hasattr(self, "_py_batt_master"):
            return (float(self._py_batt_master), float(self._py_batt_slave))
        return (100.0, 100.0)
