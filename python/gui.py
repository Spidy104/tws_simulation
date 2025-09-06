#!/usr/bin/env python3
"""
python/anim_plot.py

TWS Simulator with compact, organized controls:
- Smaller, better-positioned control elements
- Cleaner layout with logical groupings
- More space for waveform visualization
- Better visual hierarchy
"""

from __future__ import annotations
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, CheckButtons, Slider, RadioButtons
import sounddevice as sd
import logging
from typing import Optional, Tuple

# Import TWSSimulator (python wrapper that uses your compiled modules)
from tws_simulator import TWSSimulator

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anim_plot")
logger.info("anim_plot: using TWSSimulator")

# Paths
repo_root = Path(__file__).resolve().parents[1]
assets_dir = repo_root / "assets"
outputs_dir = repo_root / "outputs"
assets_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

# Matplotlib style tweaks
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "lines.antialiased": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ----------------------------
# Utility generators
# ----------------------------
def generate_sweep(start_hz: float, stop_hz: float, duration_s: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    if start_hz <= 0: start_hz = 1.0
    if stop_hz <= start_hz:
        stop_hz = start_hz + 1.0
    k = np.log(stop_hz / start_hz) / duration_s
    phase = 2 * np.pi * start_hz * (np.exp(k * t) - 1) / k
    sweep = 0.6 * np.sin(phase)
    return sweep.astype(np.float32), sweep.astype(np.float32)

def generate_preset(kind: str, duration_s: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    if kind == "white":
        L = np.random.randn(t.size) * 0.2
        R = np.random.randn(t.size) * 0.2
    elif kind == "pink":
        w = np.random.randn(t.size)
        b = np.cumsum(w)
        b = b / np.max(np.abs(b) + 1e-12) * 0.2
        L = b
        R = np.roll(b, 13)
    elif kind == "speech":
        L = np.zeros_like(t)
        R = np.zeros_like(t)
        freqs = [120, 220, 300, 400]
        i = 0
        while i < t.size:
            dur = int(sr * (0.03 + 0.07 * np.random.rand()))
            f = np.random.choice(freqs)
            end = min(t.size, i + dur)
            env = np.hanning(end - i)
            L[i:end] += 0.6 * env * np.sin(2*np.pi*f*t[i:end])
            R[i:end] += 0.6 * env * np.sin(2*np.pi*(f+10)*t[i:end])
            i = end + int(sr * (0.02 * np.random.rand()))
    elif kind == "chord":
        notes = [261.63, 329.63, 392.00]
        L = np.zeros_like(t)
        R = np.zeros_like(t)
        seg = int(sr * 0.5)
        i = 0
        j = 0
        while i < t.size:
            end = min(t.size, i + seg)
            n = notes[j % len(notes)]
            L[i:end] = 0.4 * np.sin(2*np.pi*n*t[i:end])
            R[i:end] = 0.4 * np.sin(2*np.pi*(n*1.5)*t[i:end])
            i = end
            j += 1
    else:
        L = 0.4 * np.sin(2*np.pi*440*t)
        R = 0.4 * np.sin(2*np.pi*880*t)
    return np.asarray(L, dtype=np.float32), np.asarray(R, dtype=np.float32)

# ----------------------------
# Main AnimApp - Improved Layout
# ----------------------------
class AnimApp:
    def __init__(self):
        self.sim = TWSSimulator()
        self.sample_rate = getattr(self.sim, "sample_rate", 44100)
        self.left = np.empty(0, dtype=np.float32)
        self.right = np.empty(0, dtype=np.float32)
        self.pos = 0
        self.window_seconds = 0.10
        self.window_size = max(int(self.window_seconds * self.sample_rate), 64)
        self.playing = False
        self.play_thread: Optional[threading.Thread] = None
        self.audio_event = threading.Event()

        # control defaults
        self.mode = "tone"
        self.left_freq = 440.0
        self.right_freq = 880.0
        self.sweep_start = 100.0
        self.sweep_stop = 8000.0
        self.duration = 2.0
        self.packet_loss = 0.0
        self.jitter_ms = 5.0
        self.anc_enabled = False
        self.preset_kind = "white"

        self._create_figure()
        self._create_controls()
        self._update_control_visibility()

        # Animation
        self.ani = FuncAnimation(self.fig, self._update_plot, interval=30, blit=False)

    def _create_figure(self):
        # Create figure with more space for plots
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("TWS Simulator — Real-time Audio Processing", fontsize=16, weight="bold")
        
        # Create subplot layout - more space for waveforms
        gs = self.fig.add_gridspec(4, 1, height_ratios=[2, 2, 1, 0.3], hspace=0.3)
        
        # Waveform plots
        self.ax_l = self.fig.add_subplot(gs[0])
        self.ax_r = self.fig.add_subplot(gs[1], sharex=self.ax_l)
        self.ax_batt = self.fig.add_subplot(gs[2])
        
        # Setup waveform plots
        x = np.linspace(-self.window_seconds, 0.0, self.window_size)
        self.line_l, = self.ax_l.plot(x, np.zeros(self.window_size), 'b-', lw=1.5, label="Left Channel")
        self.line_r, = self.ax_r.plot(x, np.zeros(self.window_size), 'r-', lw=1.5, label="Right Channel")
        
        self.ax_l.set_ylabel("Amplitude")
        self.ax_l.set_title("Left Channel (Master)")
        self.ax_l.legend(loc='upper right')
        self.ax_l.grid(True, alpha=0.3)
        self.ax_l.set_ylim(-1.0, 1.0)
        
        self.ax_r.set_ylabel("Amplitude")
        self.ax_r.set_xlabel("Time (s)")
        self.ax_r.set_title("Right Channel (Slave)")
        self.ax_r.legend(loc='upper right')
        self.ax_r.grid(True, alpha=0.3)
        self.ax_r.set_ylim(-1.0, 1.0)
        
        # Battery bars
        self.bars = self.ax_batt.bar(["Master", "Slave"], [85.0, 78.0], 
                                   color=["#2E86C1", "#E67E22"], alpha=0.8)
        self.ax_batt.set_ylim(0, 100)
        self.ax_batt.set_ylabel("Battery %")
        self.ax_batt.set_title("Battery Status")
        self.ax_batt.grid(True, alpha=0.3)
        self.batt_texts = [
            self.ax_batt.text(i, 5, "", ha="center", va="bottom", weight="bold", color="white")
            for i in range(2)
        ]

    def _create_controls(self):
        # Compact control dimensions
        btn_h = 0.035
        btn_w = 0.08
        text_w = 0.06
        gap_x = 0.01
        gap_y = 0.005
        
        # Control panel area (bottom of figure)
        ctrl_y_base = 0.02
        
        # Row 1: Main action buttons
        y1 = ctrl_y_base + 2*(btn_h + gap_y)
        ax_run = self.fig.add_axes([0.02, y1, btn_w, btn_h])
        ax_play = self.fig.add_axes([0.02 + btn_w + gap_x, y1, btn_w, btn_h])
        ax_stop = self.fig.add_axes([0.02 + 2*(btn_w + gap_x), y1, btn_w, btn_h])
        
        self.btn_run = Button(ax_run, "Generate")
        self.btn_play = Button(ax_play, "Play")
        self.btn_stop = Button(ax_stop, "Stop")
        
        self.btn_run.on_clicked(lambda evt: self.on_run())
        self.btn_play.on_clicked(lambda evt: self.on_play())
        self.btn_stop.on_clicked(lambda evt: self.on_stop())
        
        # Mode selection (compact radio buttons)
        ax_mode = self.fig.add_axes([0.30, y1, 0.12, btn_h])
        self.rb_mode = RadioButtons(ax_mode, ("tone", "sweep", "preset"), active=0)
        self.rb_mode.on_clicked(lambda v: self._on_mode(v))
        
        # Row 2: Frequency/Parameter controls
        y2 = ctrl_y_base + btn_h + gap_y
        
        # Tone controls
        ax_lf = self.fig.add_axes([0.45, y2, text_w, btn_h])
        ax_rf = self.fig.add_axes([0.45 + text_w + gap_x, y2, text_w, btn_h])
        self.tb_left = TextBox(ax_lf, "L Hz", initial=str(int(self.left_freq)))
        self.tb_right = TextBox(ax_rf, "R Hz", initial=str(int(self.right_freq)))
        self.tb_left.on_submit(lambda v: self._on_text_change("left_freq", v))
        self.tb_right.on_submit(lambda v: self._on_text_change("right_freq", v))
        
        # Sweep controls
        ax_ss = self.fig.add_axes([0.45, y2, text_w, btn_h])
        ax_se = self.fig.add_axes([0.45 + text_w + gap_x, y2, text_w, btn_h])
        self.tb_sstart = TextBox(ax_ss, "Start Hz", initial=str(int(self.sweep_start)))
        self.tb_sstop = TextBox(ax_se, "Stop Hz", initial=str(int(self.sweep_stop)))
        self.tb_sstart.on_submit(lambda v: self._on_text_change("sstart", v))
        self.tb_sstop.on_submit(lambda v: self._on_text_change("sstop", v))
        
        # Preset selector
        ax_preset = self.fig.add_axes([0.45, y2 - 0.02, 0.15, btn_h + 0.02])
        self.rb_preset = RadioButtons(ax_preset, ("white", "pink", "speech", "chord"))
        self.rb_preset.on_clicked(lambda v: self._on_preset(v))
        
        # Duration slider (compact)
        ax_dur = self.fig.add_axes([0.65, y2, 0.15, btn_h])
        self.sld_dur = Slider(ax_dur, "Dur(s)", valmin=0.5, valmax=10.0, 
                             valinit=self.duration, valstep=0.1, valfmt='%.1f')
        self.sld_dur.on_changed(lambda v: self._on_slider_change("duration", v))
        
        # Row 3: Network parameters
        y3 = ctrl_y_base
        
        # Packet loss and jitter
        ax_pl = self.fig.add_axes([0.02, y3, text_w, btn_h])
        ax_jt = self.fig.add_axes([0.02 + text_w + gap_x, y3, text_w, btn_h])
        self.tb_packet_loss = TextBox(ax_pl, "Loss%", initial=str(self.packet_loss))
        self.tb_jitter = TextBox(ax_jt, "Jitter", initial=str(self.jitter_ms))
        self.tb_packet_loss.on_submit(lambda v: self._on_text_change("packet_loss", v))
        self.tb_jitter.on_submit(lambda v: self._on_text_change("jitter", v))
        
        # ANC toggle (smaller checkbox)
        ax_anc = self.fig.add_axes([0.20, y3, 0.08, btn_h])
        self.chk_anc = CheckButtons(ax_anc, ["ANC"], [self.anc_enabled])
        self.chk_anc.on_clicked(lambda vals: self._on_check_change(vals))
        
        # Status indicator
        self.status_text = self.fig.text(0.85, ctrl_y_base + btn_h/2, "Ready", 
                                       ha='center', va='center', 
                                       bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor='lightgreen', alpha=0.7))

    def _on_mode(self, v: str):
        self.mode = v
        self._update_control_visibility()
        self._update_status(f"Mode: {v}")

    def _update_control_visibility(self):
        if self.mode == "tone":
            self.tb_left.ax.set_visible(True)
            self.tb_right.ax.set_visible(True)
            self.tb_sstart.ax.set_visible(False)
            self.tb_sstop.ax.set_visible(False)
            self.rb_preset.ax.set_visible(False)
        elif self.mode == "sweep":
            self.tb_left.ax.set_visible(False)
            self.tb_right.ax.set_visible(False)
            self.tb_sstart.ax.set_visible(True)
            self.tb_sstop.ax.set_visible(True)
            self.rb_preset.ax.set_visible(False)
        elif self.mode == "preset":
            self.tb_left.ax.set_visible(False)
            self.tb_right.ax.set_visible(False)
            self.tb_sstart.ax.set_visible(False)
            self.tb_sstop.ax.set_visible(False)
            self.rb_preset.ax.set_visible(True)
        self.fig.canvas.draw_idle()

    def _on_text_change(self, name: str, value: str):
        try:
            if name == "left_freq":
                self.left_freq = float(value)
            elif name == "right_freq":
                self.right_freq = float(value)
            elif name == "sstart":
                self.sweep_start = float(value)
            elif name == "sstop":
                self.sweep_stop = float(value)
            elif name == "packet_loss":
                v = float(value)
                self.packet_loss = max(0.0, min(100.0, v))
            elif name == "jitter":
                self.jitter_ms = max(0.0, float(value))
            self._update_status(f"Updated {name}")
        except Exception as e:
            self._update_status(f"Invalid {name}: {value}", error=True)
            logger.warning("Invalid input %s -> %s: %s", name, value, e)

    def _on_slider_change(self, name: str, value: float):
        if name == "duration":
            self.duration = float(value)
            self._update_status(f"Duration: {value:.1f}s")

    def _on_check_change(self, vals):
        try:
            self.anc_enabled = bool(self.chk_anc.get_status()[0])
            self._update_status(f"ANC: {'ON' if self.anc_enabled else 'OFF'}")
        except Exception:
            self.anc_enabled = False

    def _on_preset(self, value: str):
        self.preset_kind = value
        self._update_status(f"Preset: {value}")

    def _update_status(self, message: str, error: bool = False):
        color = 'lightcoral' if error else 'lightgreen'
        self.status_text.set_text(message)
        self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", 
                                     facecolor=color, alpha=0.7))

    def on_run(self):
        self._update_status("Generating...")
        logger.info("Generate/Process (mode=%s)", self.mode)
        
        # Configure simulator
        self.sim.packet_loss_prob = float(self.packet_loss) / 100.0
        self.sim.jitter_ms = float(self.jitter_ms)
        self.sim.anc_enabled = bool(self.anc_enabled)

        # Generate audio based on mode
        try:
            if self.mode == "tone":
                self.sim.generate_wav(str(assets_dir / "input.wav"), 
                                    self.left_freq, self.right_freq)
                L, R = self.sim.process_audio()
            elif self.mode == "sweep":
                L, R = generate_sweep(self.sweep_start, self.sweep_stop, self.duration, self.sample_rate)
                # Apply packet loss/jitter manually for sweep mode
                if self.packet_loss > 0.0:
                    mask = np.random.rand(L.size) >= (float(self.packet_loss) / 100.0)
                    L = L * mask
                    R = R * mask
            elif self.mode == "preset":
                L, R = generate_preset(self.preset_kind, self.duration, self.sample_rate)
                # Apply packet loss/jitter manually for preset mode
                if self.packet_loss > 0.0:
                    mask = np.random.rand(L.size) >= (float(self.packet_loss) / 100.0)
                    L = L * mask
                    R = R * mask

            self.left = np.asarray(L, dtype=np.float32)
            self.right = np.asarray(R, dtype=np.float32)
            self.pos = 0
            
            # Save outputs
            self.sim.save_output(self.left, self.right, 
                               str(outputs_dir / "left.wav"), 
                               str(outputs_dir / "right.wav"))
            
            self._update_status(f"Generated {len(self.left)} samples")
            logger.info("Generated audio: L=%d R=%d samples", len(self.left), len(self.right))
            
        except Exception as e:
            self._update_status(f"Generation failed: {str(e)[:30]}...", error=True)
            logger.exception("Generation failed")

    def on_play(self):
        if self.playing:
            self._update_status("Already playing", error=True)
            return
        if self.left.size == 0:
            self._update_status("No audio to play", error=True)
            return
        
        self._update_status("Playing...")
        self.playing = True
        self.audio_event.clear()
        self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
        self.play_thread.start()

    def on_stop(self):
        if not self.playing:
            return
        self._update_status("Stopping...")
        self.playing = False
        self.audio_event.set()
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
            self.play_thread = None
        self._update_status("Stopped")

    def _play_audio_thread(self):
        try:
            max_len = max(len(self.left), len(self.right))
            L = np.pad(self.left, (0, max_len - len(self.left)))
            R = np.pad(self.right, (0, max_len - len(self.right)))
            stereo = np.stack((L, R), axis=1)
            
            stream = sd.OutputStream(samplerate=self.sample_rate, channels=2)
            stream.start()
            chunk = 4096
            i = 0
            while i < max_len and self.playing:
                j = min(i + chunk, max_len)
                stream.write(stereo[i:j])
                i = j
                self.pos = i
                if self.audio_event.is_set():
                    break
            stream.stop()
            stream.close()
            self._update_status("Playback complete")
        except Exception as e:
            self._update_status(f"Playback error", error=True)
            logger.exception("Playback error")
        finally:
            self.playing = False
            self.pos = 0

    def _set_battery(self, master_val: float, slave_val: float):
        for rect, val, text in zip(self.bars, (master_val, slave_val), self.batt_texts):
            rect.set_height(val)
            text.set_text(f"{val:.0f}%")
            text.set_y(val/2)
            # Color coding based on battery level
            if val > 50:
                rect.set_color("#2E86C1" if rect == self.bars[0] else "#E67E22")
            elif val > 20:
                rect.set_color("#F39C12")
            else:
                rect.set_color("#E74C3C")

    def _update_plot(self, frame):
        if self.left.size == 0:
            zeros = np.zeros(self.window_size, dtype=np.float32)
            self.line_l.set_ydata(zeros)
            self.line_r.set_ydata(zeros)
        else:
            end = int(self.pos) if self.playing else min(self.window_size, len(self.left))
            end = min(end, len(self.left))
            start = max(0, end - self.window_size)
            win_l = self.left[start:end]
            win_r = self.right[start:end]
            if win_l.size < self.window_size:
                pad = self.window_size - win_l.size
                win_l = np.concatenate((np.zeros(pad, dtype=np.float32), win_l))
                win_r = np.concatenate((np.zeros(pad, dtype=np.float32), win_r))
            self.line_l.set_ydata(win_l)
            self.line_r.set_ydata(win_r)

        # Update battery display
        try:
            m, s = self.sim.get_battery_levels()
        except Exception:
            m, s = 85.0, 78.0
        self._set_battery(m, s)

        return (self.line_l, self.line_r) + tuple(self.bars)

    def show(self):
        plt.show()

# ----------------------------
# Run
# ----------------------------
def main():
    app = AnimApp()
    app.show()

if __name__ == "__main__":
    main()