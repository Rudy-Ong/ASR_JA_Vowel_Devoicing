"""
scripts/audio.py
----------------------
F0 (pitch) and RMS (intensity) feature extraction for the Gradio demos.

Recovered from the (now-removed) ``pitch_accent_viz.audio`` module — only the
two functions the demos actually use (``extract_pitch`` / ``rms_energy``) plus
their shared sanitizer. ``extract_pitch`` uses Praat via parselmouth;
``rms_energy`` uses librosa.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SR = 16_000       # 16 kHz


def _sanitize_audio(y: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip to [-1, 1].  Prevents C-level segfaults
    in parselmouth / librosa."""
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ── Pitch extraction (Parselmouth / Praat) ───────────────────────────────────

def extract_pitch(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    time_step: float = 0.005,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 contour using Praat (via parselmouth).

    Returns (times, f0_hz) — unvoiced frames are NaN.
    Returns empty arrays if audio is too short or invalid.
    """
    import parselmouth

    # ── Guard: parselmouth segfaults on empty / NaN / very short audio ───
    if y is None or len(y) < int(sr * 0.02):    # need at least 20 ms
        return np.array([]), np.array([])

    y_safe = _sanitize_audio(y)

    # All-zero audio also crashes pitch extraction
    if np.abs(y_safe).max() < 1e-10:
        return np.array([]), np.array([])

    try:
        snd = parselmouth.Sound(y_safe, sr)
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        times = pitch.xs()
        f0    = np.array([pitch.get_value_at_time(t) or np.nan for t in times])
        return times, f0
    except Exception:
        return np.array([]), np.array([])


# ── Energy / intensity ───────────────────────────────────────────────────────

def rms_energy(y: np.ndarray, hop_length: int = 160, frame_length: int = 400) -> np.ndarray:
    import librosa

    if y is None or len(y) < frame_length:
        return np.zeros(1, dtype=np.float32)
    y_safe = _sanitize_audio(y)
    return librosa.feature.rms(y=y_safe, frame_length=frame_length, hop_length=hop_length)[0]
