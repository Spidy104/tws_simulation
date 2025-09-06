#!/usr/bin/env python3
"""
python/test_cpp.py

Smoke test for the C++ extension modules.
Generates test WAVs into assets/ and prints basic diagnostics.
Saves outputs into outputs/ if applicable.
"""

import sys
from pathlib import Path
import numpy as np

# --- repo paths ---
repo_root = Path(__file__).resolve().parents[1]
build_dir = repo_root / "build"
src_dir = repo_root / "src"
assets_dir = repo_root / "assets"
outputs_dir = repo_root / "outputs"

# Ensure asset/output dirs exist
assets_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

# ensure Python can find compiled extensions in build/ or src/
if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Try imports (flexible)
def safe_import(name):
    try:
        return __import__(name)
    except Exception:
        try:
            return __import__(f"src.{name}", fromlist=[name])
        except Exception:
            return None

wg = safe_import("wav_generator")
ap = safe_import("audio_processor")
bm = safe_import("battery_model")

# --- Python fallback implementations ---
def deinterleave_samples(samples, channels):
    arr = np.asarray(samples)
    flat = arr.flatten()
    if channels == 2:
        return flat[0::2].copy(), flat[1::2].copy()
    else:
        # split into channel 0 and others (best-effort)
        return flat[0::channels].copy(), flat[1::channels].copy()

class PyAudioProcessor:
    @staticmethod
    def get_left_channel(samples, channels):
        left, _ = deinterleave_samples(samples, channels)
        return left
    @staticmethod
    def get_right_channel(samples, channels):
        _, right = deinterleave_samples(samples, channels)
        return right

class PyBattery:
    def __init__(self, capacity=50.0, initial_percent=100.0, base_drain=0.005, anc_drain=0.002):
        self.capacity = float(capacity)
        self.charge = max(0.0, min(100.0, float(initial_percent))) * 0.01 * self.capacity
        self.base = float(base_drain)
        self.anc = float(anc_drain)
    def update(self, duration_s, anc_active):
        total = duration_s * self.base + (duration_s * self.anc if anc_active else 0.0)
        self.charge = max(0.0, min(self.capacity, self.charge - total))
    def get_level(self):
        return (self.charge / self.capacity) * 100.0
    def getLevelPercent(self):
        return self.get_level()

# Pick bindings or fallbacks
WAV = wg
AP = ap if ap is not None else PyAudioProcessor
BM = bm if bm is not None else None
if BM is None:
    BM = PyBattery

def main():
    # file targets in assets/ and outputs/
    stereo_path = assets_dir / "test_stereo.wav"
    anc_path = assets_dir / "test_anc.wav"
    left_out = outputs_dir / "test_left.wav"
    right_out = outputs_dir / "test_right.wav"

    if WAV is None:
        print("ERROR: wav_generator module not found. Build it with `make` and ensure PYTHONPATH includes build/ or src/.")
        sys.exit(1)

    # instantiate WavGenerator (pybind11 binding)
    try:
        wav_gen = WAV.WavGenerator(44100, 5.0)
    except Exception as e:
        print("Failed to construct WavGenerator:", e)
        sys.exit(1)

    print("Generating stereo test WAV ->", stereo_path)
    wav_gen.generate_stereo_wav(str(stereo_path), 440.0, 880.0, 0.5)

    print("Generating ANC test WAV ->", anc_path)
    wav_gen.generate_anc_wav(str(anc_path), 440.0, 200.0, 0.5, 0.2)

    # Read back stereo file
    wdata = wav_gen.read_wav(str(stereo_path))
    samples = np.asarray(wdata.samples, dtype=np.float32)
    channels = int(wdata.channels)
    sr = int(wdata.sample_rate)
    print(f"Read WAV: channels={channels}, sample_rate={sr}, total_samples={samples.size}")

    # Deinterleave
    try:
        if ap is not None and hasattr(ap, "AudioProcessor"):
            proc = ap.AudioProcessor()
            # try different exported helper names
            if hasattr(proc, "getLeftChannel"):
                left = np.asarray(proc.getLeftChannel(samples, channels), dtype=np.float32)
                right = np.asarray(proc.getRightChannel(samples, channels), dtype=np.float32)
            elif hasattr(proc, "get_left_channel"):
                left = np.asarray(proc.get_left_channel(samples, channels), dtype=np.float32)
                right = np.asarray(proc.get_right_channel(samples, channels), dtype=np.float32)
            else:
                left, right = deinterleave_samples(samples, channels)
        else:
            left, right = deinterleave_samples(samples, channels)
            left = np.asarray(left, dtype=np.float32)
            right = np.asarray(right, dtype=np.float32)
    except Exception as e:
        print("Warning: AudioProcessor binding usage failed, falling back to Python deinterleave:", e)
        left, right = deinterleave_samples(samples, channels)
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)

    print(f"Left samples: {left.size}, Right samples: {right.size}")

    # Save per-channel outputs (as int16)
    def float_to_int16(arr):
        clipped = np.clip(arr, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(str(left_out), sr, float_to_int16(left))
        wavfile.write(str(right_out), sr, float_to_int16(right))
        print(f"Wrote left -> {left_out}, right -> {right_out}")
    except Exception as e:
        print("Failed to save channel outputs:", e)

    # Battery test
    try:
        if bm is not None and hasattr(bm, "BatteryModel"):
            batt = bm.BatteryModel(50.0, 100.0)
            if hasattr(batt, "update"):
                batt.update(5.0, True)
            lvl = None
            if hasattr(batt, "get_level"):
                lvl = batt.get_level()
            elif hasattr(batt, "getLevelPercent"):
                lvl = batt.getLevelPercent()
            else:
                lvl = getattr(batt, "get_level", lambda: float('nan'))()
            print(f"Battery (C++ binding) level after 5s (ANC active): {lvl:.2f}%")
        else:
            batt = PyBattery(50.0, 100.0)
            batt.update(5.0, True)
            print(f"Battery (py fallback) level after 5s (ANC active): {batt.get_level():.2f}%")
    except Exception as e:
        print("Battery test failed; using Python fallback:", e)
        batt = PyBattery(50.0, 100.0)
        batt.update(5.0, True)
        print(f"Battery (py fallback) level after 5s (ANC active): {batt.get_level():.2f}%")

if __name__ == "__main__":
    main()
