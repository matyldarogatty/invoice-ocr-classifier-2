# CNN + layout features — eksperymenty

## Cel rozszerzenia

Sprawdzenie, czy dodanie **znormalizowanych cech położenia linii** (bbox z docTR) do modelu **InvoiceCNN** poprawia klasyfikację semantyczną linii faktury, zwłaszcza **test macro F1**.

## Hipoteza

Obecny CNN analizuje tylko wycięty crop linii przeskalowany do 128×128. Po cropie i resize model traci informację o **bezwzględnym położeniu** linii na stronie (góra / środek / dół faktury). Late fusion z layout features powinien pomóc klasom zależnym od pozycji: `SELLER_*`, `BUYER_*`, daty, kwoty podsumowania.

Referencja — modele tekstowe z layoutem (ten sam CSV, 450 faktur):

| model_key | test macro F1 |
|-----------|---------------|
| text_svm | 0.6422 |
| text_svm_layout | **0.7460** |
| text_svm_layout_no_line_no | 0.7108 |

## Architektura `InvoiceCNNWithLayout`

Implementacja: [`src/model.py`](../src/model.py)

```text
image (1×128×128)
  → Conv blocks (jak InvoiceCNN)
  → Flatten → Linear → ReLU  → 128-d embedding

layout (10 lub 9)
  → Linear → ReLU  → 32-d embedding

concat(128, 32) → Linear → num_classes logits
```

- **`InvoiceCNN`** — bez zmian, `forward(image)`.
- **`InvoiceCNNWithLayout`** — `forward(image, layout)`.

## Cechy layoutowe

10 kolumn z [`src/layout_features.py`](../src/layout_features.py) (zakres 0–1):

`bbox_x_min_norm`, `bbox_y_min_norm`, `bbox_x_max_norm`, `bbox_y_max_norm`, `bbox_width_norm`, `bbox_height_norm`, `bbox_center_x_norm`, `bbox_center_y_norm`, `bbox_area_norm`, `line_no_norm`.

Wariant bez `line_no_norm`: 9 cech (`--exclude-line-no-feature`).

**Nie używane jako cechy:** `invoice_id`, `filename`, `label`, `semantic_name`, `template_id`.

## Dane

| Plik | Opis |
|------|------|
| `data/labels_synthetic_with_layout.csv` | 450 faktur, 28 648 linii + layout |
| `data/images_synthetic/` | cropy PNG (`filename`) |

## Nowe flagi CLI

### `train.py`

```bash
--use-layout-features          # InvoiceCNNWithLayout
--exclude-line-no-feature      # wymaga --use-layout-features
```

Bez flag layout — trening identyczny jak wcześniej (ignoruje extra kolumny w CSV).

### `run_multi_seed_experiments.py`

```bash
--with-cnn-layout              # uruchamia cnn, cnn_layout, cnn_layout_no_line_no
--layout-labels-csv data/labels_synthetic_with_layout.csv
--output-root output/cnn_layout_experiments_450
```

- **`--with-layout`** — nadal tylko eksperymenty **tekstowe** z layoutem (bez CNN).
- **`--with-cnn-layout`** — eksperymenty **CNN** (nie uruchamiać razem z `--with-layout`).

Modele CNN:

| model_key | Flagi train.py |
|-----------|----------------|
| `cnn` | (brak layout) + `--use-class-weights` |
| `cnn_layout` | `--use-layout-features --use-class-weights` |
| `cnn_layout_no_line_no` | `--use-layout-features --exclude-line-no-feature --use-class-weights` |

---

## Testy techniczne (uruchomione)

```powershell
python -m pytest tests/test_cnn_layout.py -v
```

Testy obejmują: dataset (shape, dtype, brak kolumn), forward pass modelu, smoke trening 1 epoka (CNN z/bez layout).

---

## Smoke test

```powershell
cd C:\Users\matro\PycharmProjects\AI_OCR

python src/run_multi_seed_experiments.py `
  --layout-labels-csv data/labels_synthetic_with_layout.csv `
  --images-dir data/images_synthetic `
  --output-root output/cnn_layout_smoke `
  --seeds 42 `
  --exclude-labels 10 `
  --epochs 1 `
  --batch-size 32 `
  --device auto `
  --with-cnn-layout `
  --skip-text
```

Oczekiwane katalogi: `exp_cnn_seed42/`, `exp_cnn_layout_seed42/`, `exp_cnn_layout_no_line_no_seed42/`, `multi_seed_summary.csv`.

---

## Pełny eksperyment

```powershell
python src/run_multi_seed_experiments.py `
  --layout-labels-csv data/labels_synthetic_with_layout.csv `
  --images-dir data/images_synthetic `
  --output-root output/cnn_layout_experiments_450 `
  --seeds 42 123 456 `
  --exclude-labels 10 `
  --epochs 10 `
  --batch-size 32 `
  --device auto `
  --with-cnn-layout `
  --skip-text
```

**Pełny trening multi-seed (3 seedy × 10 epok × 3 modele CNN):** _patrz sekcja Wyniki poniżej — uzupełnij po uruchomieniu._

Szacowany czas na CPU: kilka–kilkanaście godzin (28k+ linii treningowych).

---

## Odczyt wyników

Tabela podsumowująca:

```powershell
python -c "import pandas as pd; df=pd.read_csv('output/cnn_layout_experiments_450/multi_seed_summary.csv'); print(df.to_string(index=False))"
```

Ranking według macro F1:

```powershell
python -c "import pandas as pd; df=pd.read_csv('output/cnn_layout_experiments_450/multi_seed_summary.csv'); cols=[c for c in ['model_key','test_macro_f1_mean','test_macro_f1_std','test_accuracy_mean','test_accuracy_std','test_weighted_f1_mean','test_weighted_f1_std'] if c in df.columns]; print(df[cols].sort_values('test_macro_f1_mean', ascending=False).to_string(index=False))"
```

Lista modeli:

```powershell
python -c "import pandas as pd; df=pd.read_csv('output/cnn_layout_experiments_450/multi_seed_summary.csv'); print(df['model_key'].tolist())"
```

---

## Wyniki eksperymentu

### Smoke test (1 epoka, seed 42) — wstępne, **nie** zastępuje pełnego multi-seed

Uruchomiono: `output/cnn_layout_smoke/multi_seed_summary.csv`

| model_key | test_macro_f1 | test_accuracy | test_weighted_f1 |
|-----------|---------------|---------------|------------------|
| cnn | 0.6403 | 0.7521 | 0.8183 |
| cnn_layout | 0.6061 | 0.6925 | 0.7731 |
| cnn_layout_no_line_no | 0.6320 | 0.7240 | 0.7952 |

**Uwaga:** przy 1 epoce layout nie poprawił wyniku względem baseline CNN. Pełny eksperyment (10 epok × 3 seedy) jest wymagany do oceny hipotezy. Nie wyciągaj wniosków o skuteczności z smoke testu.

### Pełny multi-seed (10 epok × seedy 42, 123, 456)

_Po uruchomieniu uzupełnij tabelę z `output/cnn_layout_experiments_450/multi_seed_summary.csv`._

### Kryteria interpretacji

- **Główna metryka:** `test_macro_f1_mean`
- **Sukces:** `cnn_layout` > `cnn` + **0.01**
- **Dodatkowo:** brak dużego spadku accuracy; poprawa per-class F1 dla klas pozycyjnych (`per_class_f1.csv`)
- **Ablacja:** porównaj `cnn_layout` vs `cnn_layout_no_line_no` (wkład `line_no_norm`)

### Porównanie z text+layout

Po CNN eksperymencie porównaj z wynikami z `output/layout_text_only_450/` lub `output/layout_experiments/` (text_svm_layout ≈ 0.746).

---

## Oczekiwane artefakty

```text
output/cnn_layout_experiments_450/
├── splits_seed42/
├── exp_cnn_seed42/
├── exp_cnn_layout_seed42/
├── exp_cnn_layout_no_line_no_seed42/
├── multi_seed_per_run.csv
├── multi_seed_summary.csv
├── multi_seed_summary.json
└── per_class_f1.csv
```

---

## Ograniczenia

- Inferencja demo (`run_invoice_inference.py`) **nie obsługuje** jeszcze `InvoiceCNNWithLayout`.
- Layout pochodzi z OCR docTR (strona 0); dane syntetyczne.
- CNN może już częściowo wykorzystywać cechy wizualne cropu — wzrost może być mniejszy niż u text+layout.
- Trening na CPU jest wolny przy pełnym zbiorze.

## TODO

- [ ] Uruchomić pełny multi-seed i uzupełnić tabelę wyników
- [ ] Rozszerzyć `run_invoice_inference.py` o CNN+layout (po pozytywnych wynikach)
- [ ] Opcjonalnie: skrypt łączący wyniki CNN + text w jednej tabeli porównawczej
