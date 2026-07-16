# Phone-level3 Romaji-rev — Devoicing Detection (stratified, test split)

| model config | CER ↓ (%) | Precision ↑ (%) | Recall ↑ (%) | F1 ↑ (%) | CCDA ↑ (%) | DEO ↑ (%) |
| === |
| Batch_Size: 8; Learning_Rate: 1e-3; | 2.35 | 97.40 | 97.06 | 97.23 | 95.88 | 97.41 |
| Batch_Size: 8; Learning_Rate: 1e-4; | 8.49 | 92.61 | 91.41 | 92.01 | 89.65 | 94.12 |
| Batch_Size: 16; Learning_Rate: 1e-3; | 2.98 | 97.15 | 96.24 | 96.69 | 95.06 | 96.47 |
| Batch_Size: 16; Learning_Rate: 1e-4; | 12.50 | 88.95 | 90.00 | 89.47 | 89.88 | 94.24 |
| Batch_Size: 32; Learning_Rate: 1e-3; | 3.57 | 96.39 | 97.29 | 96.84 | 96.35 | 97.88 |
| Batch_Size: 32; Learning_Rate: 1e-4; | 18.18 | 82.53 | 86.71 | 84.57 | 84.35 | 92.47 |
