# ASR_JA_Vowel_Devoicing

Japanese speech recognition that outputs **explicit vowel-devoicing tokens**.
A Conformer-encoder / Transformer-decoder seq2seq model is trained on JSUT
basic5000 with `phone_level3`-style transcripts in which devoiced high vowels
are upper-cased (`s U k i` → し is devoiced), so recognition and devoicing
detection happen in a single pass.

Best configuration (see [doc/results.md](doc/results.md)): bs8 lr1e-3 —
**PER 2.35 %** (phone error rate; reported as "CER" before 2026-07-29),
devoicing detection **F1 97.23 %**. Character-level accuracy is reported
separately as **KER** (kana error rate — a character error rate over the
kana rendering of the phone3 output, punctuation excluded; see
`scripts/kana.py`, conventions follow
[nyosegawa/hiragana-asr](https://github.com/nyosegawa/hiragana-asr)).

## Repository layout

```
train_r3.py               training entrypoint (phone3, stratified split)
inference_r3.py           CLI inference + devoicing evaluation
eval_phone3_testset.py    batch test-set evaluation over checkpoints
gradio_demo.py            browser demo (record / pick JSUT wav, 2 checkpoints)
scripts/                  internal library + task scripts
  phone_tokenizer.py        JapaneseRomajiRevTokenizer3 (phone3 vocab)
  stratified_sampling.py    stratified manifest builder + load_splits()
  devoicing_eval.py         devoiced-vowel confusion-matrix scoring
  audio.py                  pitch / RMS extraction for the demo plots
  inference_r.py            shared decode/model-loading internals
  gradio_demo_r.py          shared demo UI helpers (plots, playhead)
  make_transcript_phone3_rev.py   preprocessing: basic5000.yaml → transcript + vocab
models/                   trained weights (git-lfs)
  train_phone3_..._bs8_lr0.001_..._stratified.pt    (phone3_bs8_lr0.001)
  train_phone3_..._bs32_lr0.001_..._stratified.pt   (phone3_bs32_lr0.001)
doc/                      results tables + tokenizer vocabularies
img/                      confusion matrices, distributions, architecture diagrams
dataset/jsut_ver1.1/basic5000/
  wav/                    ← EMPTY: download JSUT and place the 5000 wavs here
  transcript_phone3_rev.txt      training/eval transcript (devoiced I/U)
  transcript_phone3_ojt.txt      OJT-annotated reference transcript
  transcript_utf8_rev.txt        revised surface-text transcript (demo display)
  stratified_manifest.csv        train/val/test split manifest
  train_paths.txt / val_paths.txt / test_paths.txt
  basic5000.yaml                 JSUT labels (preprocessing input)
```

## Setup

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate
```

or plain pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Audio is not redistributed. Download **JSUT ver 1.1**
(https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and copy
`jsut_ver1.1/basic5000/wav/*.wav` (5000 files) into
`dataset/jsut_ver1.1/basic5000/wav/`.

## Pipeline

### 1. Preprocessing (optional — outputs are already committed)

Regenerates the phone3 transcript and tokenizer vocabulary from
`basic5000.yaml`:

```bash
python scripts/make_transcript_phone3_rev.py
```

Writes `dataset/.../transcript_phone3_rev.txt` and
`doc/tokenizer_romaji_rev_vocab.json`.

> ⚠️ The committed `transcript_phone3_rev.txt` carries a handful of manual
> corrections (5 utterances, e.g. BASIC5000_0047) on top of the rule-based
> output. Re-running the script overwrites them — use
> `git checkout dataset/.../transcript_phone3_rev.txt` to restore, or pass an
> alternative `--dst`. The stratified split manifest can **not** be rebuilt
> exactly: `scripts/stratified_sampling.py` also needs `transcript_phone.txt`
> and `transcript_romaji.txt` (from the older r1 pipeline), which are not
> committed, and regenerating from the phone3 transcripts would produce a
> *different* split. Treat the committed `stratified_manifest.csv` as
> authoritative for reproducing the published numbers.

### 2. Training

```bash
python train_r3.py
```

- Logs to [Weights & Biases](https://wandb.ai): run `wandb login` first, or
  set `WANDB_MODE=offline` (an `.env` file with `WANDB_API_KEY=…` at the repo
  root is also picked up).
- Edit the `BS` / `LR` constants in `train_r3.py` for the sweep grid; each run
  saves `models/train_phone3_<date>_bs{BS}_lr{LR}_ks19_do0.1_stratified.pt`
  and appends to `doc/results_phone3.csv`.

### 3. Inference

```bash
# single / multiple files
python inference_r3.py path/to/audio.wav --checkpoint models/train_phone3_20260624_0429_bs8_lr0.001_ks19_do0.1_stratified.pt

# JSUT test split + devoicing confusion matrix (needs the wavs)
python inference_r3.py --eval-scope test
```

Without `--checkpoint` the newest `models/train_phone3_*.pt` is used.

### 4. Test-set evaluation over all checkpoints

```bash
python eval_phone3_testset.py            # per-checkpoint PER + KER + devoicing P/R/F1
```

Confusion-matrix PNGs go to `img/`, tables to `doc/`.

### 5. Gradio demo

```bash
python gradio_demo.py
```

Record from the microphone or pick a JSUT wav, choose one of the two shipped
checkpoints (`bs8_lr0.001` / `bs32_lr0.001`) in the dropdown, and get the
phone3 transcript with devoiced vowels highlighted plus mel / pitch / RMS
plots. Launches with `share=True` (public gradio link).

## Model

| | |
|---|---|
| Encoder | Conformer × 4 (d_model 256, 4 heads, FF 1024, conv kernel 19) |
| Decoder | Transformer × 2 |
| Subsampling | Conv2d ×4 downsample |
| Audio | 16 kHz, 80 mel, n_fft 1024, hop 256 |
| Tokens | phone3 vocab (`doc/tokenizer_romaji_rev_vocab.json`), devoiced `I`/`U` |

Architecture diagram: [img/model_architecture.svg](img/model_architecture.svg)

## Publishing to GitHub

The two checkpoints (~96 MB each) are tracked with **git-lfs**:

```bash
git lfs install
git remote add origin git@github.com:<you>/ASR_JA_Vowel_Devoicing.git
git push -u origin main
```
