"""
scripts/inference_r.py
---------------------
Run transcription with a romaji-rev ConformerASR checkpoint (train_r.py).

Transcripts are whole-syllable romaji with ``pau`` for ideographic commas and a
capital vowel inside a syllable marking devoicing (し→shI, く→kU, す→sU).

Usage
-----
  # single file
  python inference_r.py audio.wav

  # multiple files
  python inference_r.py file1.wav file2.wav file3.wav

  # whole directory
  python inference_r.py --dir /path/to/wavs/

  # specify checkpoint explicitly
  python inference_r.py audio.wav --checkpoint models/train_romaji_..._stratified.pt

  # devoiced-vowel confusion matrices (i / u) on test split / full corpus
  python inference_r.py --eval-scope test

Devoiced syllables are shown in brackets, e.g.  na [kU] te → kU is devoiced.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer
from scripts import devoicing_eval as deval

# ── Audio config (must match training) ────────────────────────────────────────
SR         = 16000
N_MELS     = 80
HOP_LENGTH = 256
N_FFT      = 1_024
MAX_FRAMES = 1_024
MAX_TOKENS = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Model architecture (mirrors train_r.py) ───────────────────────────────────

class ConvSubsampling(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        freq_out = math.ceil(in_channels / 4)
        self.proj = nn.Linear(32 * freq_out, d_model)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        B, C, T2, F2 = x.shape
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, T2, C * F2))


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return 0.5 * self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.norm       = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu        = nn.GLU(dim=1)
        self.depthwise  = nn.Conv1d(d_model, d_model, kernel_size,
                                    padding=padding, groups=d_model)
        self.bn         = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)
        x = self.glu(self.pointwise1(x))
        x = self.activation(self.bn(self.depthwise(x)))
        return residual + self.dropout(self.pointwise2(x)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, kernel_size, dropout):
        super().__init__()
        self.ff1       = FeedForward(d_model, ff_dim, dropout)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv      = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2       = FeedForward(d_model, ff_dim, dropout)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + self.ff1(x)
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask)
        x = self.conv(x + y)
        return self.norm(x + self.ff2(x))


class ConformerEncoder(nn.Module):
    def __init__(self, n_mels, d_model, n_heads, ff_dim, n_layers, kernel_size, dropout):
        super().__init__()
        self.subsample = ConvSubsampling(n_mels, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_dim, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.dropout(self.subsample(x))
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ff_dim, n_layers, dropout):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, d_model)
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.layers   = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        T = tgt.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        x = self.pos_drop(self.embed(tgt))
        x = self.layers(x, memory, tgt_mask=causal_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(x)


class ConformerASR(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=256, n_heads=4, ff_dim=1024,
                 n_enc_layers=4, n_dec_layers=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(n_mels, d_model, n_heads, ff_dim,
                                        n_enc_layers, kernel_size, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, ff_dim,
                                          n_dec_layers, dropout)

    def forward(self, mel, tgt_in, tgt_pad_mask=None):
        return self.decoder(tgt_in, self.encoder(mel), tgt_pad_mask)


# ── Audio preprocessing ───────────────────────────────────────────────────────
_mel_transform = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH, n_mels=N_MELS)
_db_transform  = T.AmplitudeToDB(stype='power', top_db=80)


def apply_cmvn(mel: torch.Tensor) -> torch.Tensor:
    """Per-utterance mel mean/variance normalization over the (real, pre-pad)
    frames. Only used for models trained with CMVN (checkpoint declares
    ``cmvn=True``)."""
    mean = mel.mean(0, keepdim=True)
    std  = mel.std(0, keepdim=True).clamp_min(1e-5)
    return (mel - mean) / std


def load_mel(wav_path: Path, cmvn: bool = False) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    mel = _db_transform(_mel_transform(waveform)).squeeze(0).T  # [T, 80]
    if mel.shape[0] > MAX_FRAMES:
        mel = mel[:MAX_FRAMES]
    if cmvn:                                # normalize real frames BEFORE padding
        mel = apply_cmvn(mel)
    if mel.shape[0] < MAX_FRAMES:
        mel = torch.cat([mel, torch.zeros(MAX_FRAMES - mel.shape[0], N_MELS)])
    return mel  # [MAX_FRAMES, 80]


# ── Greedy decode ─────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(model: ConformerASR, mel: torch.Tensor,
                  tokenizer: JapaneseRomajiRevTokenizer) -> list[int]:
    memory = model.encoder(mel.unsqueeze(0).to(DEVICE))
    tgt = torch.tensor([[tokenizer.SOS_ID]], dtype=torch.long, device=DEVICE)
    for _ in range(MAX_TOKENS - 1):
        logits   = model.decoder(tgt, memory)
        next_tok = logits[:, -1, :].argmax(-1)
        # Append EOS before stopping, so a sentence-final devoiced vowel keeps
        # <eos> as its right context (C2) and CCDA can match です/ます endings.
        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
        if next_tok.item() == tokenizer.EOS_ID:
            break
    return tgt.squeeze(0).tolist()


# ── Batched greedy decode (for corpus evaluation) ─────────────────────────────
@torch.no_grad()
def greedy_decode_batch(model: ConformerASR, mel: torch.Tensor,
                        tokenizer: JapaneseRomajiRevTokenizer,
                        max_len: int = MAX_TOKENS) -> list[list[int]]:
    """Greedy-decode a batch of mels [B, MAX_FRAMES, N_MELS] → list of id lists."""
    memory = model.encoder(mel.to(DEVICE))
    B   = mel.shape[0]
    tgt = torch.full((B, 1), tokenizer.SOS_ID, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    for _ in range(max_len - 1):
        logits   = model.decoder(tgt, memory)
        next_tok = logits[:, -1, :].argmax(-1)
        next_tok = next_tok.masked_fill(finished, tokenizer.PAD_ID)
        finished |= (next_tok == tokenizer.EOS_ID)
        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return tgt.tolist()


# ── Formatting ────────────────────────────────────────────────────────────────
def format_transcript(token_ids: list[int],
                      tokenizer: JapaneseRomajiRevTokenizer) -> tuple[str, list[str]]:
    """Returns (transcript_str, devoicing_contexts).

    Devoiced tokens are bracketed in the transcript.
    devoicing_contexts lists each C1·[V]·C2 triplet found in the prediction.
    """
    devo_ids = set(tokenizer.devo_token_ids())

    # Strip PAD/SOS but keep EOS for C2 detection, then build token list
    eos_str  = "<eos>"
    clean    = [(tid, tokenizer.id2token.get(tid, "<unk>"))
                for tid in token_ids if tid not in {tokenizer.PAD_ID, tokenizer.SOS_ID}]

    parts    = []
    contexts = []
    for i, (tid, tok) in enumerate(clean):
        if tid == tokenizer.EOS_ID:
            break
        if tid in devo_ids:
            c1 = clean[i - 1][1] if i > 0 else "∅"
            c2_tok = clean[i + 1] if i + 1 < len(clean) else None
            c2 = (eos_str if c2_tok and c2_tok[0] == tokenizer.EOS_ID
                  else c2_tok[1] if c2_tok else "∅")
            parts.append(f"[{tok}]")
            contexts.append(f"{c1}·[{tok}]·{c2}")
        else:
            parts.append("," if tok == "pau" else tok)

    return " ".join(parts), contexts


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path,
               tokenizer: JapaneseRomajiRevTokenizer) -> ConformerASR:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = ConformerASR(
        vocab_size   = cfg["vocab_size"],
        n_mels       = cfg["n_mels"],
        d_model      = cfg["d_model"],
        n_heads      = cfg["n_heads"],
        ff_dim       = cfg["ff_dim"],
        n_enc_layers = cfg["n_enc_layers"],
        n_dec_layers = cfg["n_dec_layers"],
        kernel_size  = cfg["kernel_size"],
        dropout      = cfg.get("dropout", 0.0),
    ).to(DEVICE)
    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    # Whether this checkpoint was trained with CMVN. Old checkpoints lack the
    # flag → False → preprocessing is unchanged.
    model.cmvn = bool(ckpt.get("cmvn", cfg.get("cmvn", False)))
    epoch   = ckpt.get("epoch", "?")
    # 'val_CER' key kept for backward compatibility with pre-rename checkpoints
    val_per = ckpt.get("val_PER", ckpt.get("val_CER", "?"))
    devo    = ckpt.get("DEO", ckpt.get("devo_acc", "?"))
    print(f"Loaded checkpoint: {checkpoint_path.name}")
    print(f"  epoch={epoch}  val_PER={val_per}  DEO={devo}  cmvn={model.cmvn}  device={DEVICE}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def _default_checkpoint() -> Path:
    model_dir = ROOT / "models"
    candidates = sorted(model_dir.glob("train_romaji_*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No train_romaji_*.pt checkpoint found in {model_dir}")


def _default_vocab() -> Path:
    return ROOT / "doc" / "tokenizer_romaji_vocab.json"


def transcribe(wav_path: Path, model: ConformerASR,
               tokenizer: JapaneseRomajiRevTokenizer) -> tuple[str, list[str]]:
    mel = load_mel(wav_path, cmvn=getattr(model, "cmvn", False))
    ids = greedy_decode(model, mel, tokenizer)
    return format_transcript(ids, tokenizer)


# ── Corpus evaluation: devoiced-vowel confusion matrix ────────────────────────
def _load_test_ids(data_dir: Path, wav_dir: Path) -> list[str]:
    """Return the stratified test-split utterance IDs."""
    from scripts.stratified_sampling import load_splits
    manifest = data_dir / "stratified_manifest.csv"
    _, _, test_ids = load_splits(manifest, wav_dir)
    return test_ids


def _load_transcripts(data_dir: Path) -> dict[str, str]:
    """Read transcript_romaji_rev.txt → {utt_id: space-separated romaji tokens}."""
    transcripts: dict[str, str] = {}
    for line in (data_dir / "transcript_romaji_rev.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            utt, txt = line.split(":", 1)
            transcripts[utt.strip()] = txt.strip()
    return transcripts


def _load_all_ids(transcripts: dict[str, str], wav_dir: Path) -> list[str]:
    """Return every utt_id in the corpus (≈5000) that has a matching WAV, sorted."""
    return sorted(u for u in transcripts if (wav_dir / f"{u}.wav").exists())


def _run_eval(model: ConformerASR, tokenizer: JapaneseRomajiRevTokenizer,
              transcripts: dict[str, str], utt_ids: list[str], wav_dir: Path,
              model_name: str, split: str, out_img: Path, out_csv: Path,
              dual_img: Path | None = None, dual_csv: Path | None = None,
              batch_size: int = 16) -> None:
    """Decode ``utt_ids`` and build devoiced-vowel confusion matrices for one split.

    Always writes the alignment-based matrix (``out_img``/``out_csv``). When
    ``dual_img``/``dual_csv`` are given it additionally writes a matrix whose TP
    requires the CCDA context-match criterion.
    """
    print(f"Evaluating {len(utt_ids)} {split} utterances ...")

    special = {tokenizer.PAD_ID, tokenizer.SOS_ID, tokenizer.EOS_ID}
    groups  = tokenizer.vowel_groups()

    all_pred, all_ref = [], []
    cmvn = getattr(model, "cmvn", False)
    for start in range(0, len(utt_ids), batch_size):
        chunk = utt_ids[start:start + batch_size]
        mel   = torch.stack([load_mel(wav_dir / f"{u}.wav", cmvn=cmvn) for u in chunk])
        preds = greedy_decode_batch(model, mel, tokenizer)
        all_pred.extend(preds)
        all_ref.extend(tokenizer.encode(transcripts[u], add_sos=True, add_eos=True)
                       for u in chunk)
        print(f"  {min(start + batch_size, len(utt_ids))}/{len(utt_ids)}", end="\r")
    print()

    stats = deval.compute_vowel_confusion(all_pred, all_ref, special, groups)
    deval.print_confusion(stats)
    csv_path = deval.write_confusion_csv(stats, out_csv, model_name=model_name, split=split)
    img_path = deval.plot_confusion(
        stats, out_img,
        title=f"Devoiced-vowel confusion — {model_name} ({split})")
    print(f"\nConfusion CSV : {csv_path}")
    print(f"Confusion plot: {img_path}")

    if dual_img is not None and dual_csv is not None:
        ccda_stats = deval.compute_vowel_confusion_ccda(
            all_pred, all_ref, special, groups, eos_id=tokenizer.EOS_ID)
        print("\n── CCDA-scored confusion ──")
        deval.print_confusion(ccda_stats)
        dcsv = deval.write_confusion_csv(ccda_stats, dual_csv,
                                         model_name=model_name, split=split)
        dimg = deval.plot_confusion(
            ccda_stats, dual_img,
            title=f"Devoiced-vowel confusion (CCDA-scored TP) — {model_name} ({split})")
        print(f"CCDA confusion CSV : {dcsv}")
        print(f"CCDA confusion plot: {dimg}")


def evaluate(model: ConformerASR, tokenizer: JapaneseRomajiRevTokenizer,
             data_dir: Path, wav_dir: Path, model_name: str, scope: str,
             out_img: Path | None = None, out_csv: Path | None = None,
             batch_size: int = 16) -> None:
    """Run inference and build devoiced-vowel confusion matrices for ``scope``.

    ``scope`` is one of ``test`` (stratified test split, ≈500 utts),
    ``corpus`` (all utterances, ≈5000), or ``both``.
    """
    img_dir = ROOT / "img"
    csv_dir = ROOT / "doc"
    # split → (alignment-based stem, CCDA stem, utt-id loader)
    plans = {
        "test":   ("cm_vw_romaji_test_set", "cm_vw_romaji_test",
                   lambda t: [u for u in _load_test_ids(data_dir, wav_dir)
                              if u in t and (wav_dir / f"{u}.wav").exists()]),
        "corpus": ("cm_vw_romaji_corpus", "cm_vw_romaji_corpus_ccda",
                   lambda t: _load_all_ids(t, wav_dir)),
    }
    splits = ["test", "corpus"] if scope == "both" else [scope]

    if scope == "both" and (out_img or out_csv):
        print("Note: --out-img/--out-csv ignored for --eval-scope both; "
              "using fixed cm_vw_romaji_* names.")

    transcripts = _load_transcripts(data_dir)
    for split in splits:
        stem, dual_stem, id_loader = plans[split]
        utt_ids = id_loader(transcripts)
        png = out_img if (out_img and scope != "both") else img_dir / f"{stem}.png"
        csv = out_csv if (out_csv and scope != "both") else csv_dir / f"{stem}.csv"
        dual_png = img_dir / f"{dual_stem}.png"
        dual_csv = csv_dir / f"{dual_stem}.csv"
        _run_eval(model, tokenizer, transcripts, utt_ids, wav_dir,
                  model_name, split, png, csv, dual_png, dual_csv, batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Japanese ASR romaji-rev inference with vowel devoicing detection")
    parser.add_argument("wavs", nargs="*", type=Path,
                        help="WAV file(s) to transcribe")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory of WAV files to transcribe")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to romaji model checkpoint (.pt)")
    parser.add_argument("--vocab", type=Path, default=None,
                        help="Path to tokenizer_romaji_vocab.json")
    parser.add_argument("--eval-scope", choices=["test", "corpus", "both"], default=None,
                        help="Build devoiced-vowel confusion matrices (TP/FP/TN/FN for i "
                             "and u) on the stratified test split (~500 utts), the full "
                             "corpus (~5000 utts), or both")
    parser.add_argument("--eval", dest="eval_scope", action="store_const", const="test",
                        help="Back-compat alias for --eval-scope test")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "dataset" / "jsut_ver1.1" / "basic5000",
                        help="Dir with transcript_romaji_rev.txt & stratified_manifest.csv")
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="Dir of test WAVs (default: <data-dir>/wav)")
    parser.add_argument("--out-img", type=Path, default=None,
                        help="Confusion-matrix PNG path; applies only to a single "
                             "--eval-scope (ignored for 'both')")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Confusion CSV path; applies only to a single "
                             "--eval-scope (ignored for 'both')")
    args = parser.parse_args()

    vocab_path = args.vocab or _default_vocab()
    tokenizer  = JapaneseRomajiRevTokenizer.from_vocab(vocab_path)

    ckpt_path = args.checkpoint or _default_checkpoint()
    model     = load_model(ckpt_path, tokenizer)

    if args.eval_scope:
        wav_dir    = args.wav_dir or (args.data_dir / "wav")
        model_name = ckpt_path.stem
        evaluate(model, tokenizer, args.data_dir, wav_dir, model_name,
                 args.eval_scope, out_img=args.out_img, out_csv=args.out_csv)
        return

    wav_paths: list[Path] = list(args.wavs)
    if args.dir:
        if args.dir.is_file():
            wav_paths.append(args.dir)
        else:
            wav_paths += sorted(args.dir.glob("*.wav"))
    if not wav_paths:
        parser.error("Provide at least one WAV file, --dir <directory>, or --eval")

    print()
    for wav_path in wav_paths:
        if not wav_path.exists():
            print(f"[SKIP] {wav_path}  (file not found)")
            continue
        transcript, contexts = transcribe(wav_path, model, tokenizer)
        print(f"{wav_path.stem}: {transcript}")
        if contexts:
            print(f"  devoicing C1·[V]·C2: {', '.join(contexts)}")


if __name__ == "__main__":
    main()
