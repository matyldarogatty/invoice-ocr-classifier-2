# Results summary (preliminary)

This file records **one** synthetic-dataset run protocol and **preliminary** metrics. It is **not** a final thesis conclusion.

## Setup (as documented for this snapshot)

- **Data:** synthetic invoices exported to line crops + CSV; **100 documents**, **4656** labeled rows after export.
- **Quality checks:** missing labels **0**, missing text **0**, missing image files **0**.
- **Class distribution (approx.):** OTHER **3876** (~83%); **CURRENCY (id 10): 0** rows before any experiment filter.
- **Main experiments:** **11 active classes** — **CURRENCY excluded** via `--exclude-labels 10` (in memory only; `config.LABELS` unchanged).
- **Split:** document-level **70% / 15% / 15%** (`invoice_id`), seed **42** — see `output/splits_70_15_15/` (or equivalent on your machine).
- **CNN training:** 10 epochs, batch 32, **CPU** in the reference commands below (adjust `--device` as needed). **Class weights** where noted (`--use-class-weights`).
- **Metrics:** reported in **original label id** space for **active** classes only; **macro F1** averages over those classes (excluded id 10 not in the average).

## Preliminary metrics (single seed, 100 documents)

| Model | Input | Variant | val_accuracy | val_macro_f1 | val_weighted_f1 | test_accuracy | test_macro_f1 | test_weighted_f1 |
|--------|--------|---------|--------------|--------------|-------------------|---------------|---------------|------------------|
| InvoiceCNN | image | class weights, no CURRENCY | 0.9072 | **0.6324** | 0.9076 | 0.8987 | **0.4197** | 0.9005 |
| InvoiceCNN | image | class weights + train-only OTHER downsample (×3), no CURRENCY | 0.7938 | 0.5632 | 0.8345 | 0.6076 | 0.4159 | 0.7148 |
| Tfidf_LogReg | OCR text | no CURRENCY | 0.9485 | **0.6463** | 0.9343 | 0.8861 | 0.3887 | 0.8896 |
| Tfidf_LinearSVC | OCR text | no CURRENCY | **0.9485** | 0.6034 | 0.9269 | **0.9114** | 0.3901 | **0.9030** |

### Short interpretation

- **Macro F1 (validation):** highest among these runs is **TF–IDF + Logistic Regression** (0.646), closely followed by the **CNN with class weights** (0.632). Macro F1 treats each **active** class equally and is more informative than accuracy under **OTHER** dominance.
- **Accuracy (test):** highest is **TF–IDF + LinearSVC** (0.911). Accuracy is **weighted** toward frequent classes; it can look strong while rare classes are weak.
- **Train-only OTHER downsampling (CNN):** validation macro F1 **decreases** vs the CNN without downsampling in this snapshot; test accuracy **drops** sharply. This is **not** a general proof that downsampling hurts — it shows the need for more runs, tuning, and analysis (calibration, per-class confusion).

### Limitations

- **N = 100** documents: small val/test document counts; metrics are **noisy**.
- **Single seed** for splits and training; **no** confidence intervals.
- **Synthetic-only** domain; real scans and layout drift are **not** covered here.
- **CURRENCY** intentionally excluded from these rows; do not interpret as “currency is solved.”

### Next steps (experiments / documentation)

- Regenerate with **more invoices** (e.g. 300–500+), **multiple seeds**, same protocol.
- Log **git commit**, exact CLI, and paths in the thesis methods section.
- Do **not** treat this table as final thesis results without replication.
