"""
gradio_demo.py
------------------------
Gradio demo for the *phone_level3-based* romaji-rev ConformerASR model.

Same UI as ``gradio_demo_r.py`` but driven by the phone-granularity tokenizer
``JapaneseRomajiRevTokenizer3`` (vocab ``tokenizer_romaji_rev_vocab.json``,
built from JSUT ``phone_level3``). The prediction line is a space-separated
phone-token sequence (``k``, ``ky``, ``sh``, ``ts`` …) with:

  • ``cl``  – geminate / double consonant (sokuon っ)
  • ``N``   – moraic nasal ん   • ``n`` – na-row onset
  • ``,``   – rendered for a ``pau`` (ideographic comma 、)
  • ``[I]`` / ``[U]`` – devoiced high vowels, shown bracketed

All audio preprocessing, F0/RMS plotting and the playback-cursor JS are reused
from ``gradio_demo_r`` — only the tokenizer, vocab and checkpoint set differ.

This model needs a phone3 checkpoint (``train_phone3_*.pt`` in models);
with none present the line reads **No Prediction**.

Run:
    python gradio_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_TEMP_DIR", str(Path.home() / ".cache" / "gradio_tmp"))

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gradio as gr

# Reuse the inference internals — they are tokenizer-agnostic.
from scripts.inference_r import greedy_decode, format_transcript, load_model
# Reuse every audio / plot / UI-wiring helper from the romaji demo unchanged.
from scripts.gradio_demo_r import (
    to_float_mono, peak_normalize, trim_silence, mel_from_array,
    resample_16k, make_plot, list_wav_files, wav_path_for_playback,
    read_transcript, on_wav_change, PLAYHEAD_JS, WAV_DIR, TRANSCRIPT_PATH,
)
from scripts.gradio_demo_r import load_wav_file
from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer3

MODEL_DIR = ROOT / "models"
VOCAB_PATH = ROOT / "doc" / "tokenizer_romaji_rev_vocab.json"
# Checkpoints produced by a phone3 trainer (transcript_phone3_rev.txt).
CKPT_GLOB = "train_phone3_*.pt"

# Cache of {checkpoint_path: (model, tokenizer)}
_CACHE: dict[str, tuple] = {}


def list_checkpoints() -> list[str]:
    """All phone3 checkpoints, newest first."""
    ckpts = sorted(MODEL_DIR.glob(CKPT_GLOB),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ckpts]


def get_model(ckpt_path: str):
    """Load (and cache) the model + phone3 tokenizer for a checkpoint."""
    if ckpt_path not in _CACHE:
        tokenizer = JapaneseRomajiRevTokenizer3.from_vocab(VOCAB_PATH)
        model = load_model(Path(ckpt_path), tokenizer)
        _CACHE[ckpt_path] = (model, tokenizer)
    return _CACHE[ckpt_path]


# ── Prediction callback ───────────────────────────────────────────────────────

def predict(audio, wav_choice, ckpt_path):
    # Prefer a fresh recording; otherwise fall back to the selected JSUT file.
    if audio is None:
        audio = load_wav_file(wav_choice)
    if audio is None:
        return "_No audio — record something or pick a JSUT file first._", None
    if not ckpt_path:
        return "_No phone3 checkpoint found (train_phone3_*.pt)._", None
    sr, y = audio
    y_mono = to_float_mono(y)
    if y_mono.size == 0 or np.abs(y_mono).max() < 1e-6:
        return "_Audio is empty or silent._", None

    model, tokenizer = get_model(ckpt_path)

    # Match training conditions: trim silence, then peak-normalize loudness.
    # The F0/RMS plot below uses the raw recording, not this model input.
    y_model = peak_normalize(trim_silence(y_mono, sr=sr))
    mel = mel_from_array(y_model, sr, cmvn=getattr(model, "cmvn", False))
    ids = greedy_decode(model, mel, tokenizer)
    phones, contexts = format_transcript(ids, tokenizer)

    fig = make_plot(resample_16k(y_mono, sr))

    body = phones.strip() if phones.strip() else "No Prediction"
    phones_md = f"### Phones (phone_level3)\n{body}"
    if contexts:
        phones_md += f"\n\ndevoicing C1·[V]·C2: {', '.join(contexts)}"
    return phones_md, fig


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    ckpts = list_checkpoints()
    default_ckpt = ckpts[0] if ckpts else None
    wav_files = list_wav_files()

    with gr.Blocks(title="Japanese Phone ASR — Vowel Devoicing") as demo:
        gr.Markdown(
            "# Japanese Phone ASR — Vowel Devoicing (phone_level3)\n"
            "Record a phrase (or pick a JSUT file), choose a checkpoint, and press "
            "**Predict**. The line is a phone-token sequence (`cl`=geminate, "
            "`N`=ん, `n`=na-row, `,`=pause); devoiced high vowels appear bracketed "
            "(`sh [I] t e` → `I` is devoiced)."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone"], type="numpy",
                                    label="Recording (Start / Stop)")
                wav_dd = gr.Dropdown(
                    choices=wav_files, value=None,
                    label="…or pick a JSUT basic5000 file (used when not recording)",
                )
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
                phones_out = gr.Markdown(label="Phones")
        plot_out = gr.Plot(label="F0 & RMS", elem_id="fr_plot")

        wav_dd.change(on_wav_change, inputs=wav_dd,
                      outputs=[wav_player, transcript_btn, transcript_out])
        transcript_btn.click(read_transcript, inputs=wav_dd, outputs=transcript_out)

        predict_btn.click(
            predict,
            inputs=[audio_in, wav_dd, ckpt_dd],
            outputs=[phones_out, plot_out],
        ).then(
            None, None, None,
            js="() => { window.__playheadEnabled = true; "
               "window.__setupPlayhead && window.__setupPlayhead(); }",
        )

        wav_dd.change(None, None, None,
                      js="() => { window.__playheadEnabled = false; }")
        audio_in.change(None, None, None,
                        js="() => { window.__playheadEnabled = false; }")

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
