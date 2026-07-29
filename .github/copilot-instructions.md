# Copilot Instructions

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync && source .venv/bin/activate
```

Or plain pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

JSUT basic5000 audio (5000 WAVs) must be downloaded separately and placed in
`dataset/jsut_ver1.1/basic5000/wav/`. Training requires `wandb login` first,
or set `WANDB_MODE=offline`. A `.env` file at the repo root is auto-loaded for
`WANDB_API_KEY`.

## Key Commands

```bash
# Training (reads BS/LR from SWEEP_BATCH_SIZE / SWEEP_LR env vars, or defaults in train_r3.py)
python train_r3.py

# Inference — single file (defaults to newest models/train_phone3_*.pt)
python inference_r3.py path/to/audio.wav
python inference_r3.py path/to/audio.wav --checkpoint models/train_phone3_<stamp>_bs8_lr0.001_ks19_do0.1_stratified.pt

# Evaluate on JSUT test split (needs wavs)
python inference_r3.py --eval-scope test

# Batch evaluation over all checkpoints → img/ + doc/
python eval_phone3_testset.py

# Gradio demo (opens public share link)
python gradio_demo.py

# Rebuild phone3 transcript (overwrites 5 manual corrections — see README warning)
python scripts/make_transcript_phone3_rev.py

# Rebuild stratified split manifest (needs wavs)
python scripts/stratified_sampling.py
```

## Architecture

End-to-end seq2seq: **ConformerASR** (defined identically in `train_r3.py` and
`scripts/inference_r.py` — must be kept in sync if modified).

- **Encoder**: `ConvSubsampling` (Conv2d ×4 temporal downsampling) → 4 ×
  `ConformerBlock` (d=256, 4 heads, FF=1024, depthwise conv kernel=19,
  SiLU activations)
- **Decoder**: 2 × `TransformerDecoderLayer` (Pre-LN, causal mask, batch_first)
- **Audio**: 16 kHz → 80-mel log-spectrogram, n_fft=1024, hop=256,
  MAX_FRAMES=1024, MAX_TOKENS=256

Checkpoints saved as:
`models/train_phone3_<YYYYMMDD_HHMM>_bs{BS}_lr{LR}_ks19_do0.1_stratified.pt`
Tracked with **git-lfs**.

## phone_level3 / Devoicing Convention

The model outputs **phone_level3** romaji tokens, not raw text. Devoiced high
vowels are represented by **upper-case** `I` / `U` tokens; their voiced
counterparts are lower-case `i` / `u`. The two devoicing token types are the
only tokens with any upper-case character (excluding special tokens).

Tokenizer: `JapaneseRomajiRevTokenizer3` in `scripts/phone_tokenizer.py`.  
Vocab file: `doc/tokenizer_romaji_rev_vocab.json` (auto-generated on training start).  
Transcript format — each line: `UNAME:space-separated tokens`  
Example: `BASIC5000_0001:k a r e w a m i z U m i n o h I t o d e s U`

Special tokens: `<pad>=0  <sos>=1  <eos>=2  <unk>=3  <sp>=4`  
`pau` token represents an ideographic comma 、 (excluded from PER and KER scoring; rendered as 、 for display only).

## Training Details

- **Devoicing loss weight** (`DEVO_WEIGHT = 5.0`): cross-entropy loss upweights
  devoiced `I`/`U` tokens by 5× to counteract their low frequency.
- **LR schedule**: linear warmup (10 % of steps) → cosine annealing.
- **Stratified split**: 80/10/10 by phone token length quartile; manifest at
  `dataset/jsut_ver1.1/basic5000/stratified_manifest.csv` (pre-committed).
  Set `USE_STRATIFIED_SPLIT = False` in `train_r3.py` for random split.
- To sweep hyperparameters use `SWEEP_BATCH_SIZE` / `SWEEP_LR` env vars; each
  run appends to `doc/results_phone3.csv`.

## Devoicing Evaluation

`scripts/devoicing_eval.py` evaluates devoicing as a per-vowel-family binary
detection problem using **Needleman–Wunsch alignment** (`nw_align`) to handle
insertions/deletions before scoring TP/FP/TN/FN. Two scoring modes:

- `compute_vowel_confusion` — alignment-based DEO (Devoicing Error Only)
- `compute_vowel_confusion_ccda` — requires matching phonetic context (C1·V·C2)

Both are tokenizer-agnostic; callers pass `vowel_groups` from the tokenizer.

## scripts/ Module

`scripts/` is on `sys.path` at runtime (via `sys.path.insert(0, ROOT)` at the
top of entry-point scripts). Import as `from scripts.phone_tokenizer import …`.

| Module | Role |
|--------|------|
| `phone_tokenizer.py` | Tokenizer hierarchy: `JapanesePhoneTokenizer` → `JapaneseRomajiRevTokenizer3` (used by train/inference) |
| `stratified_sampling.py` | Manifest builder + `load_splits()` for train |
| `devoicing_eval.py` | NW alignment + confusion matrix + plotting |
| `inference_r.py` | Shared decode/model-loading for the romaji-rev checkpoint family |
| `gradio_demo_r.py` | Shared Gradio UI helpers (plots, playhead) |
| `audio.py` | Pitch / RMS extraction for demo plots |
| `make_transcript_phone3_rev.py` | Preprocessing: `basic5000.yaml` → phone3 transcript |
