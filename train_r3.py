"""
train_r3.py
------------------
Train a ConformerASR on the *phone_level3* romaji-rev transcripts
(transcript_phone3_rev.txt) with the phone-granularity tokenizer
JapaneseRomajiRevTokenizer3 (vocab tokenizer_romaji_rev_vocab.json).

Identical pipeline/metrics to train_r.py; only the tokenizer, transcript,
checkpoint prefix (train_phone3_*.pt — read by gradio_demo_r3.py) and results
CSV differ.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import sys, json, math, random, datetime, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make scripts/ importable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from scripts.phone_tokenizer import JapaneseRomajiRevTokenizer3
from scripts.stratified_sampling import load_splits
from scripts.kana import compute_kana_error_rate
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import libtmux
import wandb

# Load WANDB_API_KEY from .claude/.env
_env_path = Path(__file__).resolve().parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if '=' in _line and not _line.startswith('#'):
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())
wandb.login()

ROOT = Path(__file__).resolve().parent

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')

# Every batch has the same shape (MAX_FRAMES × N_MELS), so cudnn autotuning
# pays off without re-benchmarking per batch.
torch.backends.cudnn.benchmark = True

# This machine runs several training jobs (ours + other users') across the
# same GPUs/cores at once; torch defaults to one intra-op thread per core,
# which oversubscribes the box and makes iteration time jittery. Cap it.
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '6')))
torch.set_num_interop_threads(1)

# Optional: allow TF32 matmuls (Ampere+).  Off by default to keep numerics
# identical to the published runs; enable with ALLOW_TF32=1.  (Conv layers
# already use TF32 by default via cudnn — this flag only affects matmuls.)
if os.environ.get('ALLOW_TF32', '0') == '1':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    print('TF32 matmul: ON (ALLOW_TF32=1)')
else:
    print('TF32 matmul: off (default; ALLOW_TF32=1 to enable)')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / 'dataset' / 'jsut_ver1.1' / 'basic5000'
WAV_DIR   = DATA_DIR / 'wav'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR = ROOT / 'doc'

# ── Audio config ──────────────────────────────────────────────────────────────
SR         = 16000
N_MELS     = 80
HOP_LENGTH = 256
N_FFT      = 1024
MAX_FRAMES = 1024
MAX_TOKENS = 256

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE   = int(os.environ.get('SWEEP_BATCH_SIZE', 16))
NUM_EPOCHS   = 100
LR           = float(os.environ.get('SWEEP_LR', 1e-3))
WEIGHT_DECAY = 1e-2
RESUME_FROM  = os.environ.get('RESUME_FROM')

# ── Model config ──────────────────────────────────────────────────────────────
D_MODEL      = 256
N_HEADS      = 4
N_ENC_LAYERS = 4
N_DEC_LAYERS = 2
FF_DIM       = 1024
DROPOUT      = 0.1
KERNEL_SIZE  = 19
DEVO_WEIGHT  = float(os.environ.get('SWEEP_DEVO_WEIGHT', 5))

# Optional: mask zero-padded mel frames out of the encoder self-attention.
# Off by default so runs reproduce the published pipeline exactly; set
# ENC_PADDING_MASK=1 to restrict attention to real frames (changes training).
USE_ENC_PADDING_MASK = os.environ.get('ENC_PADDING_MASK', '0') == '1'

# ── Stratified split config ───────────────────────────────────────────────────
USE_STRATIFIED_SPLIT = True   # set True after: python scripts/stratified_sampling.py
MANIFEST_PATH = DATA_DIR / 'stratified_manifest.csv'


# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
tokenizer = JapaneseRomajiRevTokenizer3(DATA_DIR / 'transcript_phone3_rev.txt')
tokenizer.save(TOKENIZER_DIR / 'tokenizer_romaji_rev_vocab.json')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Devoicing tokens: {len(tokenizer.devo_token_ids())}')


# ── 2. Transcripts & split ────────────────────────────────────────────────────
def load_transcripts(path: Path) -> dict[str, str]:
    data = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if ':' not in line:
            continue
        utt_id, text = line.split(':', 1)
        data[utt_id.strip()] = text.strip()
    return data

romaji_transcripts = load_transcripts(DATA_DIR / 'transcript_phone3_rev.txt')
all_ids = sorted(k for k in romaji_transcripts if (WAV_DIR / f'{k}.wav').exists())
print(f'Total utterances with wav: {len(all_ids)}')

if USE_STRATIFIED_SPLIT:
    train_ids, val_ids, test_ids = load_splits(MANIFEST_PATH, WAV_DIR)
    print(f'Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)} StratifiedSampling')
else:
    train_ids, tmp_ids = train_test_split(all_ids, test_size=0.20, random_state=SEED)
    val_ids, test_ids  = train_test_split(tmp_ids,  test_size=0.50, random_state=SEED)
    print(f'[random]     Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')


# ── 3. Dataset & DataLoader ───────────────────────────────────────────────────
mel_transform = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH, n_mels=N_MELS)
db_transform  = T.AmplitudeToDB(stype='power', top_db=80)
freq_mask     = T.FrequencyMasking(freq_mask_param=15)
time_mask     = T.TimeMasking(time_mask_param=70)


class JSUTDataset(Dataset):
    def __init__(self, utt_ids, transcripts, wav_dir, tokenizer):
        self.utt_ids     = utt_ids
        self.transcripts = transcripts
        self.wav_dir     = wav_dir
        self.tokenizer   = tokenizer

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id   = self.utt_ids[idx]
        waveform, sr = torchaudio.load(str(self.wav_dir / f'{utt_id}.wav'))
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        mel = db_transform(mel_transform(waveform)).squeeze(0).T  # [T, 80]
        if mel.shape[0] > MAX_FRAMES:
            mel = mel[:MAX_FRAMES]
        mel_len = mel.shape[0]
        if mel.shape[0] < MAX_FRAMES:
            mel = torch.cat([mel, torch.zeros(MAX_FRAMES - mel.shape[0], N_MELS)])

        ids = self.tokenizer.encode(self.transcripts[utt_id], add_sos=True, add_eos=True)
        ids = ids[:MAX_TOKENS]
        tok_len = len(ids)
        if len(ids) < MAX_TOKENS:
            ids = ids + [self.tokenizer.PAD_ID] * (MAX_TOKENS - len(ids))

        return {'mel': mel, 'mel_len': mel_len, 'tokens':  torch.tensor(ids, dtype=torch.long), 'tok_len': tok_len, 'utt_id':  utt_id,}


def make_loader(utt_ids, shuffle=False):
    ds = JSUTDataset(utt_ids, romaji_transcripts, WAV_DIR, tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=(DEVICE.type == 'cuda'),
                      persistent_workers=True, prefetch_factor=2)

train_loader = make_loader(train_ids, shuffle=True)
val_loader   = make_loader(val_ids)
test_loader  = make_loader(test_ids)


# ── 4. Model Architecture ─────────────────────────────────────────────────────
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
        self.ff1      = FeedForward(d_model, ff_dim, dropout)
        self.attn     = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
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

    def forward(self, x, key_padding_mask=None):
        x = self.dropout(self.subsample(x))
        for layer in self.layers:
            x = layer(x, key_padding_mask)
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

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        T = tgt.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
        x = self.pos_drop(self.embed(tgt))
        x = self.layers(x, memory, tgt_mask=causal_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(x)


class ConformerASR(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=256, n_heads=4, ff_dim=1024,
                 n_enc_layers=4, n_dec_layers=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(n_mels, d_model, n_heads, ff_dim,
                                        n_enc_layers, kernel_size, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, ff_dim,
                                          n_dec_layers, dropout)

    def forward(self, mel, tgt_in, tgt_pad_mask=None, enc_pad_mask=None):
        return self.decoder(tgt_in, self.encoder(mel, enc_pad_mask), tgt_pad_mask,
                            memory_key_padding_mask=enc_pad_mask)


model = ConformerASR(
    vocab_size=tokenizer.vocab_size, n_mels=N_MELS, d_model=D_MODEL,
    n_heads=N_HEADS, ff_dim=FF_DIM, n_enc_layers=N_ENC_LAYERS,
    n_dec_layers=N_DEC_LAYERS, kernel_size=KERNEL_SIZE, dropout=DROPOUT,
).to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M')


# ── 5. Loss & Optimizer ───────────────────────────────────────────────────────
_weights = torch.ones(tokenizer.vocab_size, device=DEVICE)
for _tid in tokenizer.devo_token_ids():
    _weights[_tid] = DEVO_WEIGHT
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID, weight=_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps  = NUM_EPOCHS * len(train_loader)
warmup_steps = total_steps // 10

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── 6. Metrics ────────────────────────────────────────────────────────────────
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    return dp[n]


def compute_per(pred_ids, ref_ids):
    """Phone Error Rate over the phone3 romaji-rev token sequences.

    The Levenshtein distance runs over phone-level tokens (``k``, ``sh``,
    ``cl``, ``I``/``U`` …), so this is a phone(-token) error rate — historically
    logged as "CER" in this repo.  ``pau`` and ``<eos>`` (plus
    ``<pad>``/``<sos>``) are NOT counted: they are stripped from both hypothesis
    and reference before the edit-distance, so PER measures only the spoken
    phones.  See ``scripts.kana.compute_kana_error_rate`` for the
    character-level metric (KER, kana error rate).
    """
    special = {tokenizer.PAD_ID, tokenizer.SOS_ID, tokenizer.EOS_ID, tokenizer.pau_id}
    total_dist = total_ref = 0
    for pred, ref in zip(pred_ids, ref_ids):
        pred = [t for t in pred if t not in special]
        ref  = [t for t in ref  if t not in special]
        total_dist += edit_distance(pred, ref)
        total_ref  += len(ref)
    return total_dist / max(1, total_ref) * 100


def compute_devoicing_deo(pred_ids, ref_ids):
    """
    Devoicing Event Occurances (DEO) is a metrics to calculate devoicing detection occurances

    Input:
    pred_ids: List of predicted token ID sequences (with SOS/EOS/PAD)
    ref_ids: List of reference token ID sequences (with SOS/EOS/PAD)

    Output:
    - deo: devoicing event occurances is a percentage of correctly predicted in utterances/sentences
    - correct: Count of correctly predicted devoicing events in number
    - total: Count of devoicing events in reference in number

    """
    special  = {tokenizer.PAD_ID, tokenizer.SOS_ID, tokenizer.EOS_ID}
    devo_ids = set(tokenizer.devo_token_ids())
    correct = total = 0
    for pred, ref in zip(pred_ids, ref_ids):
        pred_clean = [t for t in pred if t not in special]
        ref_clean  = [t for t in ref  if t not in special]
        total += sum(1 for t in ref_clean if t in devo_ids)
        for d in devo_ids:
            correct += min(ref_clean.count(d), pred_clean.count(d))
    deo = correct / total * 100 if total > 0 else float('nan')
    return deo, correct, total


def compute_devoicing_ccda(pred_ids, ref_ids):
    """Context-Conditioned Devoicing Accuracy (CCDA)
    A custom metric to check if devoiced vowels are predicted in the correct phonetic context C1-V-C2.
    To be counted as correct, the predicted devoiced vowel must match a reference devoiced vowel and
    also correct predicting just-left (C1) and right (C2) neighbours of a devoiced vowel.
    In case of devoicing vowel like "です。" and "ます。", <EOS> is count as C2

    Input:
    - pred_ids: List of predicted token ID sequences (with SOS/EOS/PAD)
    - ref_ids: List of reference token ID sequences (with SOS/EOS/PAD)

    Output:
    - ccda: context conditioned devoicing accuracy is a percentage of correctly predicted devoiced vowels with left and right consonants also predicted correctly
    - correct: Count of correctly predicted ccda in number
    - total: Count of devoiced vowels in reference

    """
    skip     = {tokenizer.PAD_ID, tokenizer.SOS_ID}   # keep EOS for C2
    devo_ids = set(tokenizer.devo_token_ids())
    correct  = total = 0
    for pred, ref in zip(pred_ids, ref_ids):
        pred_clean = [t for t in pred if t not in skip]
        ref_clean  = [t for t in ref  if t not in skip]

        pred_triplets: dict[tuple, int] = {}
        for i, tok in enumerate(pred_clean):
            if tok in devo_ids:
                c1  = pred_clean[i - 1] if i > 0 else None
                c2  = pred_clean[i + 1] if i + 1 < len(pred_clean) else None
                key = (c1, tok, c2)
                pred_triplets[key] = pred_triplets.get(key, 0) + 1

        for i, tok in enumerate(ref_clean):
            if tok in devo_ids:
                total += 1
                c1  = ref_clean[i - 1] if i > 0 else None
                c2  = ref_clean[i + 1] if i + 1 < len(ref_clean) else None
                key = (c1, tok, c2)
                if pred_triplets.get(key, 0) > 0:
                    correct += 1
                    pred_triplets[key] -= 1

    ccda = correct / total * 100 if total > 0 else float('nan')
    return ccda, correct, total

def _align(ref, hyp):
    """Levenshtein alignment of two token-key lists → list of (ri, hi) index
    pairs; a gap (insertion / deletion) carries ``None`` on the missing side.

    Used by the devoicing-detection metric, which aligns on the *devoicing-folded*
    key (I↔i, U↔u collapsed) — the same alignment PER uses — so a high-vowel slot
    lines up regardless of whether it is devoiced; the devoicing decision is then
    read from the original token ids.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            pairs.append((i-1, None)); i -= 1
        else:
            pairs.append((None, j-1)); j -= 1
    pairs.reverse()
    return pairs


def compute_devoicing_detection(pred_ids, ref_ids):
    """Binary devoicing **detection** scored per high-vowel slot (Precision/Recall/F1).

    Each high vowel (i / u, voiced or devoiced) is one slot; the binary decision
    is *devoiced* (I / U) vs *voiced* (i / u). Hypothesis and reference are
    Levenshtein-aligned on the devoicing-folded key (I↔i, U↔u) so a slot lines up
    regardless of its devoicing, and a True Positive is an aligned slot that is
    the **same** high vowel and **devoiced in both**:

      recall    = TP / (all reference devoiced vowels)   → catches under-devoicing
      precision = TP / (all predicted  devoiced vowels)  → catches over-devoicing
      f1        = harmonic mean of precision and recall

    Unlike CCDA this does not require the C1/C2 neighbours to be correct — it
    isolates the devoicing decision itself. CCDA, reported alongside, adds the
    C1-V-C2 context condition and is therefore the stricter measure; the
    (recall − CCDA) gap is correct devoicings sitting in mis-recognized context.

    Input:
    - pred_ids: List of predicted token ID sequences (with SOS/EOS/PAD)
    - ref_ids:  List of reference token ID sequences (with SOS/EOS/PAD)

    Output:
    - f1, precision, recall  (percentages)
    - tp, fp, fn  (counts; fp = over-devoicings, fn = missed devoicings)
    """
    special  = {tokenizer.PAD_ID, tokenizer.SOS_ID, tokenizer.EOS_ID, tokenizer.pau_id}
    id2tok   = tokenizer.id2token
    devo_set = {'I', 'U'}

    def fold(t):                                  # devoicing-blind vowel key
        s = id2tok.get(t, '')
        return 'i' if s == 'I' else 'u' if s == 'U' else s

    tp = ref_dev = pred_dev = 0
    for pred, ref in zip(pred_ids, ref_ids):
        r = [t for t in ref  if t not in special]
        h = [t for t in pred if t not in special]
        ref_dev  += sum(1 for t in r if id2tok.get(t, '') in devo_set)
        pred_dev += sum(1 for t in h if id2tok.get(t, '') in devo_set)
        rfold = [fold(t) for t in r]
        hfold = [fold(t) for t in h]
        for ri, hi in _align(rfold, hfold):
            if ri is None or hi is None:
                continue
            if (rfold[ri] in ('i', 'u') and rfold[ri] == hfold[hi]
                    and id2tok.get(r[ri], '') in devo_set
                    and id2tok.get(h[hi], '') in devo_set):
                tp += 1

    fp, fn = pred_dev - tp, ref_dev - tp
    precision = tp / pred_dev * 100 if pred_dev else float('nan')
    recall    = tp / ref_dev  * 100 if ref_dev  else float('nan')
    if math.isnan(precision) or math.isnan(recall):
        f1 = float('nan')
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, tp, fp, fn


# ── 7. Greedy Decode ──────────────────────────────────────────────────────────
def _build_enc_padding_mask(mel_len) -> torch.Tensor:
    """key_padding_mask [B, T'] (True = padded) for the ×4-subsampled encoder
    output, built from per-utterance mel frame counts.  Inputs are always
    padded to MAX_FRAMES, so T' = ceil(MAX_FRAMES / 4)."""
    out_len = (MAX_FRAMES + 3) // 4
    keep = (mel_len.to(DEVICE) + 3) // 4          # ceil(len / 4) real frames
    return torch.arange(out_len, device=DEVICE).unsqueeze(0) >= keep.unsqueeze(1)


@torch.no_grad()
def greedy_decode_batch(model, mel, max_len=MAX_TOKENS, mel_len=None):
    model.eval()
    B = mel.shape[0]
    _m = model.module if isinstance(model, nn.DataParallel) else model
    enc_mask = (_build_enc_padding_mask(mel_len)
                if USE_ENC_PADDING_MASK and mel_len is not None else None)
    memory = _m.encoder(mel.to(DEVICE), enc_mask)

    tgt      = torch.full((B, 1), tokenizer.SOS_ID, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)

    for _ in range(max_len - 1):
        logits   = _m.decoder(tgt, memory, memory_key_padding_mask=enc_mask)
        next_tok = logits[:, -1, :].argmax(-1)
        next_tok = next_tok.masked_fill(finished, tokenizer.PAD_ID)
        finished |= (next_tok == tokenizer.EOS_ID)
        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
        if finished.all():
            break

    return tgt.tolist()


# ── 8. tmux session + DataParallel ───────────────────────────────────────────
_tmux_server = libtmux.Server()
_session_name = "ja_devoicing_training_phone3"
try:
    _session = _tmux_server.sessions.get(session_name=_session_name)
    print(f"Attached to existing tmux session: {_session_name}")
except Exception:
    _session = _tmux_server.new_session(session_name=_session_name)
    print(f"Created new tmux session: {_session_name}")

def _probe_dataparallel(dp_model) -> bool:
    """Run a tiny forward pass to force DataParallel's cross-GPU replicate/NCCL
    broadcast. Returns True if it succeeds, False if the multi-GPU path is
    broken (e.g. NCCL 'unhandled system error' from a host driver mismatch)."""
    try:
        with torch.no_grad():
            dummy_mel = torch.zeros(2, MAX_FRAMES, N_MELS, device=DEVICE)
            dummy_tgt = torch.zeros(2, 2, dtype=torch.long, device=DEVICE)
            dp_model(dummy_mel, dummy_tgt)
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        print(f"[WARN] Multi-GPU probe failed: {exc}")
        return False


if torch.cuda.device_count() > 1:
    _dp = nn.DataParallel(model)
    if _probe_dataparallel(_dp):
        model = _dp
        print(f"DataParallel across {torch.cuda.device_count()} GPUs: "
              + str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    else:
        # NCCL / multi-GPU broadcast is unusable — fall back to single GPU so
        # training can still proceed. `model` stays the un-wrapped module on DEVICE.
        del _dp
        torch.cuda.empty_cache()
        print(f"[WARN] Falling back to single-GPU training on {DEVICE}.")
else:
    print(f"Single GPU: {DEVICE}")

print(f'Encoder padding mask: '
      f'{"ON (ENC_PADDING_MASK=1)" if USE_ENC_PADDING_MASK else "off (default; ENC_PADDING_MASK=1 to enable)"}')


# ── 9. Train & Eval Functions ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, global_step=0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        mel    = batch['mel'].to(DEVICE)
        mel    = mel.permute(0, 2, 1)
        mel    = freq_mask(mel)
        mel    = time_mask(mel)
        mel    = mel.permute(0, 2, 1)
        tokens = batch['tokens'].to(DEVICE)
        tgt_in  = tokens[:, :-1]
        tgt_out = tokens[:, 1:]
        pad_mask = (tgt_in == tokenizer.PAD_ID)
        enc_mask = _build_enc_padding_mask(batch['mel_len']) if USE_ENC_PADDING_MASK else None

        logits = model(mel, tgt_in, tgt_pad_mask=pad_mask, enc_pad_mask=enc_mask)
        loss   = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        global_step += 1
        if global_step % 25 == 0:
            wandb.log({'train/loss': loss.item(),'train/lr':   scheduler.get_last_lr()[0]}, step=global_step)
    return total_loss / len(loader), global_step


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_pred, all_ref = [], []

    for batch in tqdm(loader, desc='eval', leave=False):
        mel    = batch['mel'].to(DEVICE)
        tokens = batch['tokens'].to(DEVICE)
        tgt_in  = tokens[:, :-1]
        tgt_out = tokens[:, 1:]
        pad_mask = (tgt_in == tokenizer.PAD_ID)
        enc_mask = _build_enc_padding_mask(batch['mel_len']) if USE_ENC_PADDING_MASK else None

        logits = model(mel, tgt_in, tgt_pad_mask=pad_mask, enc_pad_mask=enc_mask)
        loss   = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
        total_loss += loss.item()

        all_pred.extend(greedy_decode_batch(model, mel, mel_len=batch['mel_len']))
        all_ref.extend(tokens.tolist())

    per = compute_per(all_pred, all_ref)
    ker = compute_kana_error_rate(all_pred, all_ref, tokenizer)
    DEO, deo_correct, devo_total = compute_devoicing_deo(all_pred, all_ref)
    CCDA, ccda_correct, _        = compute_devoicing_ccda(all_pred, all_ref)
    F1, P, R, tp, fp, fn         = compute_devoicing_detection(all_pred, all_ref)
    return {
        'loss': total_loss / len(loader), 'per': per, 'ker': ker,
        'f1': F1, 'precision': P, 'recall': R, 'tp': tp, 'fp': fp, 'fn': fn,
        'ccda': CCDA, 'ccda_correct': ccda_correct,
        'deo': DEO, 'deo_correct': deo_correct,            # retained: logging only
        'devo_total': devo_total,
    }


# ── 10. tmux pane + log file ──────────────────────────────────────────────────
RUN_TAG    = datetime.datetime.now().strftime('%Y%m%d_%H%M')
NAME_SUFFIX = os.environ.get('SWEEP_NAME_SUFFIX', 'stratified')
model_name = f'train_phone3_{RUN_TAG}_bs{BATCH_SIZE}_lr{LR}_ks{KERNEL_SIZE}_do{DROPOUT}' + (f'_{NAME_SUFFIX}' if NAME_SUFFIX else '')

wandb.init(
    project='ja_devoicing_asr',
    name=model_name,
    config={
        'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'd_model': D_MODEL, 'n_heads': N_HEADS,
        'n_enc_layers': N_ENC_LAYERS, 'n_dec_layers': N_DEC_LAYERS,
        'ff_dim': FF_DIM, 'dropout': DROPOUT, 'kernel_size': KERNEL_SIZE,
        'warmup_steps': warmup_steps, 'total_steps': total_steps,
        'devo_weight': DEVO_WEIGHT, 'tokenizer': 'romaji_rev_phone3',
    },
)

_log_path  = MODEL_DIR / f'{model_name}_train.log'
_log_file  = open(_log_path, 'w', buffering=1)

try:
    _pane = _session.active_window.active_pane
    _pane.send_keys(f'tail -f {_log_path}', enter=True)
    print(f'Streaming epoch logs to tmux pane  →  {_log_path}')
except Exception:
    _pane = None
    print(f'Logs written to {_log_path}')


def log(msg: str) -> None:
    print(msg)
    _log_file.write(msg + '\n')


# ── 11. Resume from previous checkpoint ──────────────────────────────────────
start_epoch = 1
best_per    = float('inf')
global_step = 0

_resume_path = Path(RESUME_FROM) if RESUME_FROM else MODEL_DIR / f'{model_name}.pt'
if _resume_path.exists():
    _ckpt = torch.load(_resume_path, map_location=DEVICE)
    _m = model.module if isinstance(model, nn.DataParallel) else model
    _state = _ckpt['model_state']
    if any(k.startswith('module.') for k in _state):
        _state = {k.removeprefix('module.'): v for k, v in _state.items()}
    _m.load_state_dict(_state)
    optimizer.load_state_dict(_ckpt['optimizer_state'])
    if 'scheduler_state' in _ckpt:
        scheduler.load_state_dict(_ckpt['scheduler_state'])
    start_epoch = _ckpt['epoch'] + 1
    # 'val_CER' key kept for backward compatibility with pre-rename checkpoints
    best_per    = float(_ckpt.get('val_PER', _ckpt.get('val_CER', 'inf')).replace('%', ''))
    global_step = _ckpt['epoch'] * len(train_loader)
    log(f'Resumed from checkpoint: {_resume_path.name}  epoch={_ckpt["epoch"]} | best_PER={best_per:.4f}%')
elif RESUME_FROM:
    raise FileNotFoundError(f'RESUME_FROM checkpoint not found: {RESUME_FROM}')
else:
    log('No checkpoint found, training from scratch.')


# ── 12. Training Loop ─────────────────────────────────────────────────────────
best_val_result = None

log(f'=== Training start: {model_name}  epochs={NUM_EPOCHS}  batch={BATCH_SIZE} ===')

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    train_loss, global_step = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, global_step)
    m = eval_epoch(model, val_loader, criterion)
    val_per = m['per']

    log(
        f'Epoch {epoch}/{NUM_EPOCHS} | '
        f'train_loss={train_loss:.4f} | val_loss={m["loss"]:.4f} | '
        f'val_PER={val_per:.2f}% | '
        f'KER={m["ker"]:.2f}% | '
        f'devoF1={m["f1"]:.2f}% (P={m["precision"]:.2f} R={m["recall"]:.2f}, '
        f'TP={m["tp"]} FP={m["fp"]} FN={m["fn"]}) | '
        f'CCDA={m["ccda"]:.2f}% ({m["ccda_correct"]}/{m["devo_total"]}) | '
        f'DEO={m["deo"]:.2f}% ({m["deo_correct"]}/{m["devo_total"]})'
    )

    wandb.log({
        'train_loss@epoch': train_loss,
        'val_loss@epoch':   m['loss'],
        'val_per@epoch':    val_per,
        'ker@epoch':        m['ker'],
        'devoF1@epoch':         None if math.isnan(m['f1'])        else m['f1'],
        'devo_precision@epoch': None if math.isnan(m['precision']) else m['precision'],
        'devo_recall@epoch':    None if math.isnan(m['recall'])    else m['recall'],
        'CCDA@epoch':           None if math.isnan(m['ccda'])      else m['ccda'],
        'DEO@epoch':            None if math.isnan(m['deo'])       else m['deo'],
    }, step=global_step)

    if val_per < best_per:
        best_per  = val_per
        best_val_result = {'epoch': epoch, **m}
        ckpt_path = MODEL_DIR / f'{model_name}.pt'
        _m_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch':            epoch,
            'model_state':      _m_save.state_dict(),
            'optimizer_state':  optimizer.state_dict(),
            'scheduler_state':  scheduler.state_dict(),
            'val_PER':         f'{val_per:.2f}%',
            'KER':             f'{m["ker"]:.2f}%',
            'devo_F1':         f'{m["f1"]:.2f}%',
            'devo_P':          f'{m["precision"]:.2f}%',
            'devo_R':          f'{m["recall"]:.2f}%',
            'CCDA':            f'{m["ccda"]:.2f}%',
            'DEO':             f'{m["deo"]:.2f}%',
            'config': {
                'vocab_size': tokenizer.vocab_size, 'n_mels': N_MELS,
                'd_model': D_MODEL, 'n_heads': N_HEADS, 'ff_dim': FF_DIM,
                'n_enc_layers': N_ENC_LAYERS, 'n_dec_layers': N_DEC_LAYERS,
                'kernel_size': KERNEL_SIZE, 'dropout': DROPOUT,
            },
        }, ckpt_path)
        log(f'  → Saved best checkpoint: {ckpt_path.name}')

_test_ckpt_path = MODEL_DIR / f'{model_name}.pt'
if not _test_ckpt_path.exists():
    _test_ckpt_path = _resume_path
    log(f'No new best checkpoint saved this run; using {_test_ckpt_path.name} for test eval.')

log('\nTraining complete.')
_log_file.close()
wandb.finish()


# ── 13. Test Evaluation ───────────────────────────────────────────────────────
ckpt = torch.load(_test_ckpt_path, map_location=DEVICE)
_m   = model.module if isinstance(model, nn.DataParallel) else model
_state = ckpt['model_state']
if any(k.startswith('module.') for k in _state):
    _state = {k.removeprefix('module.'): v for k, v in _state.items()}
_m.load_state_dict(_state)

if best_val_result is None:
    # RESUME_FROM a checkpoint whose epoch was never beaten this run (no new
    # best found) — best_val_result was never populated, so the val split
    # never gets a CSV row. Re-evaluate val with the checkpoint actually used
    # for test, so its real validation numbers still get recorded.
    best_val_result = {'epoch': ckpt['epoch'], **eval_epoch(model, val_loader, criterion)}

tm = eval_epoch(model, test_loader, criterion)
test_per = tm['per']
print(f'\n=== Test Results ===')
print(f'Test PER    : {test_per:.2f}%')
print(f'Test KER    : {tm["ker"]:.2f}%')
print(f'Devo P/R/F1 : {tm["precision"]:.2f}% / {tm["recall"]:.2f}% / {tm["f1"]:.2f}%  '
      f'(TP={tm["tp"]} FP={tm["fp"]} FN={tm["fn"]})')
print(f'CCDA        : {tm["ccda"]:.2f}%  ({tm["ccda_correct"]}/{tm["devo_total"]})')
print(f'DEO         : {tm["deo"]:.2f}%  ({tm["deo_correct"]}/{tm["devo_total"]})')


# ── 14. Save Results CSV ──────────────────────────────────────────────────────
csv_path  = ROOT / 'doc' / 'results_phone3.csv'
today     = datetime.date.today().isoformat()

def _pct(x, nd):
    return None if (isinstance(x, float) and math.isnan(x)) else round(x, nd)

def _row(split, epoch, d, nd):
    return {
        'model_name': model_name, 'date': today, 'split': split, 'epoch': epoch,
        'PER':     _pct(d['per'], nd),
        'KER':     _pct(d['ker'], nd),
        'devo_P':  _pct(d['precision'], nd),
        'devo_R':  _pct(d['recall'], nd),
        'devo_F1': _pct(d['f1'], nd),
        'TP': d['tp'], 'FP': d['fp'], 'FN': d['fn'],
        'CCDA':         _pct(d['ccda'], nd),
        'CCDA_correct': d['ccda_correct'],
        'DEO':          _pct(d['deo'], nd),
        'DEO_correct':  d['deo_correct'],
        'devo_total':   d['devo_total'],
    }

all_rows = []
if best_val_result:
    all_rows.append(_row('val', best_val_result['epoch'], best_val_result, 2))
all_rows.append(_row('test', ckpt['epoch'], tm, 5))

fieldnames = ['model_name', 'date', 'split', 'epoch', 'PER', 'KER',
              'devo_P', 'devo_R', 'devo_F1', 'TP', 'FP', 'FN',
              'CCDA', 'CCDA_correct', 'DEO', 'DEO_correct', 'devo_total']
if csv_path.exists():
    with open(csv_path, newline='', encoding='utf-8') as f:
        existing_header = next(csv.reader(f), [])
    write_header = existing_header != fieldnames
else:
    write_header = True
with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerows(all_rows)
print(f'Results saved → {csv_path}')
try:
    print(pd.read_csv(csv_path).tail(5).to_string(index=False))
except Exception as e:
    print(f'(skipping results preview, could not read {csv_path}: {e})')


# ── 15. Quick Inference Demo ──────────────────────────────────────────────────
@torch.no_grad()
def transcribe_one(wav_path: Path) -> str:
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    mel = db_transform(mel_transform(waveform)).squeeze(0).T
    if mel.shape[0] > MAX_FRAMES:
        mel = mel[:MAX_FRAMES]
    elif mel.shape[0] < MAX_FRAMES:
        mel = torch.cat([mel, torch.zeros(MAX_FRAMES - mel.shape[0], N_MELS)])
    ids  = greedy_decode_batch(model, mel.unsqueeze(0).to(DEVICE))[0]
    return tokenizer.decode(ids)

import random
number  = random.randint(0, len(test_ids) - 1)
demo_id  = test_ids[number]
demo_wav = WAV_DIR / f'{demo_id}.wav'
demo_ref = romaji_transcripts[demo_id]
demo_pred_str = transcribe_one(demo_wav)
print(f'\nID  : {demo_id}')
print(f'REF : {tokenizer.render_pau(demo_ref)}')
print(f'PRED: {tokenizer.render_pau(demo_pred_str)}')

# encode both for single-utterance metric check
demo_ref_ids  = [tokenizer.encode(demo_ref,      add_sos=True, add_eos=True)]
demo_pred_ids = [tokenizer.encode(demo_pred_str, add_sos=True, add_eos=True)]
_f1, _p, _r, _tp, _fp, _fn = compute_devoicing_detection(demo_pred_ids, demo_ref_ids)
_ccda, _cor_pos, _tot      = compute_devoicing_ccda(demo_pred_ids, demo_ref_ids)
_deo, _deo_cor, _          = compute_devoicing_deo(demo_pred_ids, demo_ref_ids)
print(f'Devo P/R/F1 : {_p:.2f}% / {_r:.2f}% / {_f1:.2f}%  (TP={_tp} FP={_fp} FN={_fn})')
print(f'CCDA : {_ccda:.2f}%  ({_cor_pos}/{_tot})')
print(f'DEO  : {_deo:.2f}%  ({_deo_cor}/{_tot})')
