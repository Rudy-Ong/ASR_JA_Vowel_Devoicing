"""
scripts/gradio_demo_r.py
-----------------------
Gradio demo for the Japanese *romaji* ConformerASR model (train_r.py).

Record from the microphone (Start / Stop) or pick a JSUT file, choose a romaji
checkpoint, press **Predict**: the app shows one line — **romaji** — with vowel
devoicing surfaced as a capital vowel inside a bracketed syllable
(``na [kU] te`` → ``kU`` is devoiced). Below the text it plots the F0 (pitch)
and RMS (intensity) contours of the recording, mirroring
``pitch-accent-viz analyze``.

This model only produces romaji (there is no romaji→phone/hiragana converter),
so when it returns nothing the line reads **No Prediction**.

Run:
    python gradio_demo_r.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "GRADIO_TEMP_DIR", str(Path.home() / ".cache" / "gradio_tmp")
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

# Reuse the inference internals — do not re-implement the model.
from scripts.inference_r import (
    SR, N_MELS, MAX_FRAMES,
    _mel_transform, _db_transform,
    apply_cmvn,
    greedy_decode, format_transcript, load_model,
)
from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer
# F0 / RMS extraction (recovered from the removed pitch-accent-viz toolkit).
from scripts.audio import extract_pitch, rms_energy

MODEL_DIR = ROOT / "models"
VOCAB_PATH = ROOT / "doc" / "tokenizer_romaji_vocab.json"
# JSUT directory
WAV_DIR = ROOT / "dataset" / "jsut_ver1.1" / "basic5000" / "wav"
# Original JSUT transcripts, one "BASIC5000_xxxx:text" line per utterance.
TRANSCRIPT_PATH = ROOT / "dataset" / "jsut_ver1.1" / "basic5000" / "transcript_utf8_rev.txt"

# Cache of {checkpoint_path: (model, tokenizer)}
_CACHE: dict[str, tuple] = {}

# The romaji checkpoint this demo was built around — used as the UI default when
# present, otherwise the newest romaji checkpoint is used.
DEFAULT_CKPT_NAME = "train_romaji_20260619_1903_bs16_lr0.001_do0.1_stratified.pt"


def list_wav_files() -> list[str]:
    """All JSUT basic5000 wav filenames, sorted."""
    if not WAV_DIR.is_dir():
        return []
    return sorted(p.name for p in WAV_DIR.glob("*.wav"))


def list_checkpoints() -> list[str]:
    """All romaji checkpoints, newest first."""
    ckpts = sorted(MODEL_DIR.glob("train_romaji_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ckpts]


def get_model(ckpt_path: str):
    """Load (and cache) the model + tokenizer for a checkpoint."""
    if ckpt_path not in _CACHE:
        tokenizer = JapaneseRomajiRevTokenizer.from_vocab(VOCAB_PATH)
        model = load_model(Path(ckpt_path), tokenizer)
        _CACHE[ckpt_path] = (model, tokenizer)
    return _CACHE[ckpt_path]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def to_float_mono(y: np.ndarray) -> np.ndarray:
    """Gradio mic audio → float32 mono in [-1, 1]."""
    y = np.asarray(y)
    if y.ndim == 2:                       # (samples, channels) → mono
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    return y.astype(np.float32)


def peak_normalize(y_mono: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale the waveform so its peak amplitude matches ``target_peak``.

    Training data (JSUT) is recorded at a consistent, fairly hot level, while
    live mic input is usually much quieter. Because the mel uses an *absolute*
    AmplitudeToDB reference, that loudness gap shifts every feature value and
    hurts the model. Peak-normalizing brings mic input back in line."""
    peak = float(np.abs(y_mono).max())
    if peak < 1e-6:
        return y_mono
    return (y_mono * (target_peak / peak)).astype(np.float32)


def trim_silence(y_mono: np.ndarray, thresh_ratio: float = 0.02,
                 margin_s: float = 0.05, sr: int = SR) -> np.ndarray:
    """Trim leading/trailing near-silence the model never saw in trimmed JSUT.

    Keeps everything above ``thresh_ratio`` of the peak, plus a small margin so
    onsets/offsets aren't clipped. Returns the input unchanged if it's silent."""
    peak = float(np.abs(y_mono).max())
    if peak < 1e-6:
        return y_mono
    above = np.abs(y_mono) > (thresh_ratio * peak)
    if not above.any():
        return y_mono
    idx = np.flatnonzero(above)
    margin = int(margin_s * sr)
    start = max(0, idx[0] - margin)
    end = min(len(y_mono), idx[-1] + 1 + margin)
    return y_mono[start:end]


def mel_from_array(y_mono: np.ndarray, sr: int, cmvn: bool = False) -> torch.Tensor:
    """numpy mono audio → padded mel [MAX_FRAMES, N_MELS] (mirrors inference_r.load_mel)."""
    wav = torch.from_numpy(y_mono).unsqueeze(0)        # [1, T]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    mel = _db_transform(_mel_transform(wav)).squeeze(0).T   # [T, N_MELS]
    if mel.shape[0] > MAX_FRAMES:
        mel = mel[:MAX_FRAMES]
    if cmvn:                                # match training (CMVN models only)
        mel = apply_cmvn(mel)
    if mel.shape[0] < MAX_FRAMES:
        mel = torch.cat([mel, torch.zeros(MAX_FRAMES - mel.shape[0], N_MELS)])
    return mel


def resample_16k(y_mono: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono audio to 16 kHz float32 for F0 / RMS analysis."""
    if sr == SR:
        return y_mono
    wav = torch.from_numpy(y_mono).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0).numpy()


def make_plot(y16: np.ndarray) -> go.Figure:
    """F0 (pitch) and RMS (intensity) contours, styled after `analyze`.

    Returns an interactive Plotly figure (zoom / pan / hover) with two stacked,
    time-aligned subplots. ``shapes[0]`` is a hidden full-height vertical line —
    the playback cursor that client-side JS moves via ``Plotly.relayout`` while
    the audio plays (see the ``demo.load`` JS in ``build_ui``)."""
    dur = len(y16) / SR
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("F0 vs Time", "Intensity vs Time"))

    # F0 — unvoiced frames are NaN; connectgaps=False renders them as gaps.
    times, f0 = extract_pitch(y16, SR)
    if len(times) > 0:
        fig.add_trace(
            go.Scatter(x=times, y=f0, name="F0", connectgaps=False,
                       line=dict(color="steelblue", width=1.5)),
            row=1, col=1,
        )

    # RMS energy — rms_energy returns no times, build a matching axis.
    rms = rms_energy(y16)
    if len(rms) > 1:
        t_rms = np.linspace(0, dur, num=len(rms))
        fig.add_trace(
            go.Scatter(x=t_rms, y=rms, name="RMS",
                       line=dict(color="coral", width=1.5)),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="F0 (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="RMS Energy", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", range=[0, max(dur, 0.01)], row=2, col=1)

    # Playback cursor — hidden until JS turns it on during playback.
    fig.add_shape(type="line", x0=0, x1=0, yref="paper", y0=0, y1=1,
                  line=dict(color="red", width=2), visible=False)

    fig.update_layout(title="Recording analysis", showlegend=False,
                      dragmode=False, margin=dict(t=60))
    return fig


# ── Prediction callback ───────────────────────────────────────────────────────

def load_wav_file(name: str) -> tuple[int, np.ndarray] | None:
    """Load a JSUT wav by filename into Gradio's ``(sr, samples)`` form."""
    if not name:
        return None
    path = WAV_DIR / name
    if not path.is_file():
        return None
    wav, sr = torchaudio.load(str(path))      # [channels, T] float32
    return sr, wav.T.numpy()                   # (T, channels) like gr.Audio numpy


def wav_path_for_playback(name: str) -> str | None:
    """Filepath of the selected JSUT wav, for the playback widget (or None)."""
    if not name:
        return None
    path = WAV_DIR / name
    return str(path) if path.is_file() else None


def read_transcript(name: str) -> str:
    """Original JSUT transcript text for the selected wav (no ``BASIC5000_xxxx:``).

    ``name`` is a wav filename like ``BASIC5000_1293.wav``; we strip the
    extension and return the text after the colon on the matching
    ``BASIC5000_1293:…`` line of ``transcript_utf8.txt``.
    """
    if not name:
        return "Pick a JSUT file first."
    utt_id = Path(name).stem                  # BASIC5000_1293.wav → BASIC5000_1293
    prefix = f"{utt_id}:"
    if not TRANSCRIPT_PATH.is_file():
        return f"Transcript file not found: {TRANSCRIPT_PATH}"
    with TRANSCRIPT_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return f"No transcript found for {utt_id}."


def on_wav_change(name: str):
    """Sync the playback widget, transcript button, and transcript box to the pick.

    Returns updates for ``[wav_player, transcript_btn, transcript_out]``: the
    button is enabled only when a JSUT file is selected (disabled for recording),
    and the transcript box is cleared whenever the selection is removed.
    """
    enabled = bool(name)
    transcript = gr.update() if enabled else ""   # leave text until pressed; clear on deselect
    return wav_path_for_playback(name), gr.update(interactive=enabled), transcript


def predict(audio, wav_choice, ckpt_path):
    # Prefer a fresh recording; otherwise fall back to the selected JSUT file.
    if audio is None:
        audio = load_wav_file(wav_choice)
    if audio is None:
        return "_No audio — record something or pick a JSUT file first._", None
    sr, y = audio
    y_mono = to_float_mono(y)
    if y_mono.size == 0 or np.abs(y_mono).max() < 1e-6:
        return "_Audio is empty or silent._", None

    model, tokenizer = get_model(ckpt_path)

    # Match training conditions: trim silence, then peak-normalize loudness.
    # Done only on the model input — the F0/RMS plot below uses the raw recording.
    # CMVN (when the checkpoint declares it) further removes level/coloration.
    y_model = peak_normalize(trim_silence(y_mono, sr=sr))
    mel = mel_from_array(y_model, sr, cmvn=getattr(model, "cmvn", False))
    ids = greedy_decode(model, mel, tokenizer)
    romaji, contexts = format_transcript(ids, tokenizer)

    y16 = resample_16k(y_mono, sr)
    fig = make_plot(y16)

    # Only romaji is produced by this model; when it's empty, say so explicitly.
    body = romaji.strip() if romaji.strip() else "No Prediction"
    romaji_md = f"### Romaji\n{body}"
    if contexts:
        romaji_md += f"\n\ndevoicing C1·[V]·C2: {', '.join(contexts)}"
    return romaji_md, fig


# ── Playback cursor (client-side) ─────────────────────────────────────────────
# Installed once via demo.load. Moves the plot's vertical line (shapes[0]) in
# sync with whichever <audio> element is playing, but only after Predict flips
# window.__playheadEnabled on. Clicking the plot seeks the active audio.
PLAYHEAD_JS = """
() => {
  window.__playheadEnabled = false;
  window.__gd = null;
  window.__activeAudio = null;

  // Follow playback: move shapes[0] to the audio's current time.
  const onTime = (e) => {
    if (!(e.target instanceof HTMLAudioElement)) return;
    window.__activeAudio = e.target;
    const gd = window.__gd;
    if (!window.__playheadEnabled || !gd || !window.Plotly) return;
    const t = e.target.currentTime;
    window.Plotly.relayout(gd, {
      'shapes[0].x0': t, 'shapes[0].x1': t, 'shapes[0].visible': true,
    });
  };
  document.addEventListener('timeupdate', onTime, true);
  document.addEventListener('play', (e) => {
    if (e.target instanceof HTMLAudioElement) window.__activeAudio = e.target;
  }, true);

  // (Re)bind to the freshly rendered Plotly div and wire click-to-seek.
  window.__setupPlayhead = function () {
    let tries = 0;
    const bind = () => {
      const gd = document.querySelector('#fr_plot .js-plotly-plot');
      if (!gd || !gd._fullLayout) {
        if (tries++ < 40) requestAnimationFrame(bind);
        return;
      }
      window.__gd = gd;
      if (!gd.__seekBound) {
        gd.__seekBound = true;
        gd.addEventListener('click', (ev) => {
          const xa = gd._fullLayout.xaxis;
          const a = window.__activeAudio;
          if (!xa || !a) return;
          const rect = gd.getBoundingClientRect();
          let t = xa.p2l(ev.clientX - rect.left - xa._offset);
          const dur = a.duration || t;
          t = Math.max(0, Math.min(t, dur));
          a.currentTime = t;
          if (window.__playheadEnabled && window.Plotly) {
            window.Plotly.relayout(gd, {
              'shapes[0].x0': t, 'shapes[0].x1': t, 'shapes[0].visible': true,
            });
          }
        });
      }
    };
    bind();
  };
}
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    ckpts = list_checkpoints()
    preferred = [c for c in ckpts if Path(c).name == DEFAULT_CKPT_NAME]
    default_ckpt = preferred[0] if preferred else (ckpts[0] if ckpts else None)
    wav_files = list_wav_files()

    with gr.Blocks(title="Japanese Romaji ASR — Vowel Devoicing") as demo:
        gr.Markdown(
            "# Japanese Romaji ASR — Vowel Devoicing\n"
            "Record a phrase (or pick a JSUT file), choose a checkpoint, and press "
            "**Predict**. Devoiced vowels appear as a capital vowel inside a "
            "bracketed syllable in the romaji line (e.g. `na [kU] te` → `kU` is "
            "devoiced)."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone"], type="numpy",
                                    label="Recording (Start / Stop)")
                wav_dd = gr.Dropdown(
                    choices=wav_files, value=None,
                    label="…or pick a JSUT basic5000 file (used when not recording)",
                )
                # Only meaningful for a JSUT pick — disabled until one is chosen.
                transcript_btn = gr.Button("Read Original JSUT Transcript",
                                           interactive=False)
                wav_player = gr.Audio(type="filepath", label="Selected JSUT file",
                                      interactive=False)
                ckpt_dd = gr.Dropdown(
                    choices=ckpts, value=default_ckpt, label="Model checkpoint",
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                transcript_out = gr.Textbox(label="Original JSUT transcript",
                                            interactive=False)
                romaji_out = gr.Markdown(label="Romaji")
        plot_out = gr.Plot(label="F0 & RMS", elem_id="fr_plot")

        # Picking a file loads the player and enables the transcript button;
        # clearing the pick disables it and wipes any shown transcript.
        wav_dd.change(on_wav_change, inputs=wav_dd,
                      outputs=[wav_player, transcript_btn, transcript_out])
        transcript_btn.click(read_transcript, inputs=wav_dd, outputs=transcript_out)

        predict_btn.click(
            predict,
            inputs=[audio_in, wav_dd, ckpt_dd],
            outputs=[romaji_out, plot_out],
        ).then(
            None, None, None,
            js="() => { window.__playheadEnabled = true; "
               "window.__setupPlayhead && window.__setupPlayhead(); }",
        )

        # A new, un-predicted clip must show no cursor until Predict runs again.
        wav_dd.change(None, None, None,
                      js="() => { window.__playheadEnabled = false; }")
        audio_in.change(None, None, None,
                        js="() => { window.__playheadEnabled = false; }")

        # Install the playback-cursor listeners once, on page load.
        demo.load(None, None, None, js=PLAYHEAD_JS)
    return demo


if __name__ == "__main__":
    build_ui().launch(
        share=True,
        allowed_paths=[
            str(WAV_DIR),
            str(TRANSCRIPT_PATH),
        ]
    )
