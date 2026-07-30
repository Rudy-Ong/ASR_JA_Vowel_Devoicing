> **Metric naming (2026-07-29):** what was reported as "CER" below is an edit
> distance over tokenizer *tokens*, not characters — it is now called **PER**
> (phone/phone-token error rate). A true character-level metric, **KER**
> (kana error rate: edit distance over the kana rendering of the phone3
> sequences, punctuation excluded — see `scripts/kana.py`, conventions follow
> github.com/nyosegawa/hiragana-asr), was added the same day; historical rows
> predate it.

# Phoneme Recognition Results (stratified, test split)

| model config | PER (%) | DEO (%) | CCDA (%) |
|---|---|---|---|
| Batch_Size: 8; Learning_Rate: 1e-3; | 7.86 | 93.83 | 89.86 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 15.14 | 90.70 | 83.38 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 19.75 | 91.85 | 83.18 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 7.66 | 93.83 | 91.01 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 8.29 | 92.79 | 89.55 |
| Batch_Size: 32; Learning_Rate: 1e-4; | 25.99 | 89.55 | 76.80 |

# Romaji Recognition Results (stratified, test split)

(units of the edit distance are romaji *syllable* tokens — tokenizer_romaji_vocab.json)

| model config | PER ↓ (%) | DEO ↑ (%) | CCDA ↑ (%) |
|---|---|---|---|
| Batch_Size: 8; Learning_Rate: 1e-3; | 6.25 | 95.49 | 90.37 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 12.70 | 92.32 | 83.90 |
| Batch_Size: 8; Learning_Rate: 1e-5; | 88.34 | 41.83 | 2.20 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 6.09 | 96.10 | 89.88 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 16.05 | 91.71 | 82.32 |
| Batch_Size: 16; Learning_Rate: 1e-5; | 86.57 | 42.20 | 1.83 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 6.82 | 95.85 | 89.51 |
| Batch_Size: 32; Learning_Rate: 1e-4; | 21.69 | 89.63 | 76.71 |
| Batch_Size: 32; Learning_Rate: 1e-5; | 88.68 | 39.02 | 1.71 |

# Phone-level3 Romaji-rev Recognition Results (stratified, test split)

| model config | PER ↓ (%) | DEO ↑ (%) | CCDA ↑ (%) |
|---|---|---|---|
| Batch_Size: 8; Learning_Rate: 1e-3; | 2.35 | 97.41 | 95.88 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 8.50 | 94.00 | 89.53 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 2.98 | 96.47 | 95.06 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 12.50 | 94.24 | 89.88 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 3.57 | 97.88 | 96.35 |
| Batch_Size: 32; Learning_Rate: 1e-4; | 18.18 | 92.47 | 84.35 |

# Phone-level3 Romaji-rev — Devoicing Detection (stratified, test split)

| model config | PER ↓ (%) | Precision ↑ (%) | Recall ↑ (%) | F1 ↑ (%) | CCDA ↑ (%) | DEO ↑ (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Batch_Size: 8; Learning_Rate: 1e-3; | **2.35** | **97.40** | **97.06** | **97.23** | 95.88 | 97.41 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 8.49 | 92.61 | 91.41 | 92.01 | 89.65 | 94.12 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 2.98 | 97.15 | 96.24 | 96.69 | 95.06 | 96.47 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 12.50 | 88.95 | 90.00 | 89.47 | 89.88 | 94.24 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 3.57 | 96.39 | 97.29 | 96.84 | **96.35** | **97.88** |
| Batch_Size: 32; Learning_Rate: 1e-4; | 18.18 | 82.53 | 86.71 | 84.57 | 84.35 | 92.47 |

| model config | PER ↓ (%) | F1 ↑ (%) | CCDA ↑ (%) |
| --- | --- | --- | --- |
| Batch_Size: 8; Learning_Rate: 1e-3; | **2.35** | **97.23** | 95.88 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 8.49 | 92.01 | 89.65 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 2.98 | 96.69 | 95.06 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 12.50 | 89.47 | 89.88 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 3.57 | 96.84 | **96.35** |
| Batch_Size: 32; Learning_Rate: 1e-4; | 18.18 | 84.57 | 84.35 |

| model config | PER ↓ (%) | CCDA ↑ (%) |
| --- | --- | --- |
| BS: 8; LR: 1e-3; | **2.35** | 95.88 |
| BS: 8; LR: 1e-4; | 8.49 | | 89.65 |
| BS: 16; LR: 1e-3; | 2.98  | 95.06 |
| BS: 16; LR: 1e-4; | 12.50 | 89.88 |
| BS: 32; LR: 1e-3; | 3.57  | **96.35** |
| BS: 32; LR: 1e-4; | 18.18 | 84.35 |

# Reproduction (2026-07-29, RTX 5090, torch 2.13+cu130, seed 42)

Checkpoint `train_phone3_20260729_1404_bs8_lr0.001_ks19_do0.1_stratified.pt`,
best val epoch 82/100, evaluated on the stratified test split (500 utts):

| model config | PER ↓ (%) | KER ↓ (%) | Precision ↑ (%) | Recall ↑ (%) | F1 ↑ (%) | CCDA ↑ (%) | DEO ↑ (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Batch_Size: 8; Learning_Rate: 1e-3; | 2.49 | 3.59 | 97.05 | 96.82 | 96.94 | 95.06 | 96.82 |

Confirms the claimed bs8/lr1e-3 results (PER 2.35 %, F1 97.23 %) within
run-to-run variance. KER is the new character-level companion metric (kana
error rate): Levenshtein over the kana rendering with `pau`/punctuation
excluded (conventions follow github.com/nyosegawa/hiragana-asr); devoiced
I/U fold to い/う, so it is deliberately devoicing-blind.