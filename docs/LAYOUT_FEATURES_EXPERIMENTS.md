# Layout / bbox features — eksperymenty tekstowe

## Cel rozszerzenia

Sprawdzenie, czy dodanie **znormalizowanych cech położenia linii** (bbox z docTR) do modeli tekstowych TF-IDF poprawia klasyfikację semantyczną linii faktury, zwłaszcza **test macro F1**.

Na tym etapie zaimplementowano wyłąznie wariant **TF-IDF + klasyfikator liniowy + layout**. **CNN + layout** pozostaje przyszłym rozszerzeniem.

## Dodane cechy layoutowe

Dla każdej linii OCR (envelope wszystkich słów w linii, współrzędne z docTR `word.geometry`, już znormalizowane 0–1 względem strony):

| Kolumna | Opis |
|---------|------|
| `bbox_x_min_norm` | min(x) słów linii |
| `bbox_y_min_norm` | min(y) słów linii |
| `bbox_x_max_norm` | max(x) słów linii |
| `bbox_y_max_norm` | max(y) słów linii |
| `bbox_width_norm` | x_max − x_min |
| `bbox_height_norm` | y_max − y_min |
| `bbox_center_x_norm` | (x_min + x_max) / 2 |
| `bbox_center_y_norm` | (y_min + y_max) / 2 |
| `bbox_area_norm` | width × height |
| `line_no_norm` | line_idx / max(1, total_lines − 1) na stronie |

**Nie używane jako cechy modelu:** `invoice_id`, `filename`, `crop_path`, `semantic_name`, `label`, `template_id`.

Implementacja: [`src/layout_features.py`](../src/layout_features.py).

## Nowe pliki danych

| Plik | Opis |
|------|------|
| `data/labels_synthetic_with_layout.csv` | Główny CSV z 5 dotychczasowymi kolumnami + 10 layout |
| `data/labels_synthetic_with_layout_review.csv` | Review CSV z layout (opcjonalnie) |

Stary `data/labels_synthetic.csv` **nie jest nadpisywany**.

## Nowe flagi CLI

### `export_synthetic_to_labels.py`

```bash
--include-layout-features
```

Gdy aktywna: zapisuje layout columns; domyślny output → `data/labels_synthetic_with_layout.csv`.

### `train_text_baseline.py`

```bash
--use-layout-features          # TF-IDF + StandardScaler(layout) + klasyfikator
--exclude-line-no-feature      # wymaga --use-layout-features; pomija line_no_norm
```

Bez `--use-layout-features` skrypt działa jak wcześniej (ignoruje dodatkowe kolumny w CSV).

### `run_multi_seed_experiments.py`

```bash
--with-layout
--layout-labels-csv data/labels_synthetic_with_layout.csv
--output-root output/layout_experiments
```

Gdy `--with-layout`: uruchamia baseline text + 4 warianty layout na layout CSV; **pomija CNN**; output domyślnie → `output/layout_experiments/`.

Modele layout:

- `text_logreg_layout`
- `text_svm_layout`
- `text_logreg_layout_no_line_no`
- `text_svm_layout_no_line_no`

### `validate_layout_csv.py`

Walidacja CSV po eksporcie (kolumny, NaN, zakres 0–1, kolejność bbox).

---

## Testy techniczne (uruchomione)

Komenda:

```powershell
python -m pytest tests/test_layout_features.py tests/test_text_layout_training.py -v
```

| Test | Wynik |
|------|-------|
| `test_layout_features.py` (9 testów) | **PASSED** |
| `test_train_text_baseline_ignores_extra_columns` | **PASSED** |
| `test_train_text_baseline_with_layout_features` | **PASSED** |
| `test_train_text_baseline_layout_without_line_no` | **PASSED** |
| `test_real_layout_csv_validation` | skipped do czasu pełnego CSV |

Dodatkowo wykonano smoke-check:

1. Eksport 3 faktur z `--include-layout-features` → 94 wiersze, walidacja OK
2. Split dokumentowy seed=42
3. Trening `text_svm_layout` → `output/layout_smoke/exp_text_svm_layout_seed42/metrics.json`

**Uwaga:** metryki ze smoke-run (3 faktury) **nie są reprezentatywne** dla oceny jakości modelu.

---

## Pełne uczenie i metryki — NIE uruchomione w tej iteracji

Pełny multi-seed (100 faktur × 6 modeli tekstowych × 3 seedy) **nie został uruchomiony** w ramach tej implementacji (czasochłonny eksport OCR + trening). Poniżej instrukcja ręczna.

---

## How to run full layout experiments

### Krok 1 — wygenerowanie CSV z layout features

```powershell
cd C:\Users\matro\PycharmProjects\AI_OCR
python src/export_synthetic_to_labels.py --include-layout-features
```

Opcjonalnie jawna ścieżka:

```powershell
python src/export_synthetic_to_labels.py `
  --include-layout-features `
  --csv-path data/labels_synthetic_with_layout.csv `
  --review-csv-path data/labels_synthetic_with_layout_review.csv
```

**Oczekiwany wynik:** plik `data/labels_synthetic_with_layout.csv` z wierszami OCR + 10 kolumn layout.

Po pełnym eksporcie (2026-05-29): **610 faktur, 28 648 linii**, walidacja OK.

> **Uwaga:** stary `data/labels_synthetic.csv` ma tylko **4656 wierszy** (wcześniejszy, mniejszy manifest). Dla uczciwego porównania używaj baseline text uruchamianego przez `--with-layout` na **tym samym** layout CSV — nie porównuj wprost z `output/multi_seed/` opartym o stary plik.

### Krok 2 — walidacja CSV

```powershell
python src/validate_layout_csv.py data/labels_synthetic_with_layout.csv
```

**Oczekiwany wynik:**

```text
Layout columns present: 10/10
NaN count: 0
Out of range [0,1]: 0
Invalid bbox order: 0
OK: True
```

Porównanie liczby wierszy:

```powershell
python -c "import pandas as pd; a=pd.read_csv('data/labels_synthetic.csv'); b=pd.read_csv('data/labels_synthetic_with_layout.csv'); print(len(a), len(b))"
```

Jeśli liczby się różnią, ponownie uruchom baseline text na **tym samym** layout CSV dla uczciwego porównania.

### Krok 3 — split dokumentowy 70/15/15 (pojedynczy seed, test)

```powershell
python src/create_splits.py `
  --labels-csv data/labels_synthetic_with_layout.csv `
  --output-dir output/layout_experiments/splits_seed42 `
  --seed 42
```

### Krok 4 — pojedynczy model testowy (TF-IDF + LinearSVC + layout)

```powershell
python src/train_text_baseline.py `
  --train-csv output/layout_experiments/splits_seed42/train.csv `
  --val-csv output/layout_experiments/splits_seed42/val.csv `
  --test-csv output/layout_experiments/splits_seed42/test.csv `
  --output-dir output/layout_experiments/exp_text_svm_layout_seed42 `
  --model linear_svc `
  --seed 42 `
  --exclude-labels 10 `
  --use-layout-features
```

**Metryki:** `output/layout_experiments/exp_text_svm_layout_seed42/metrics.json`

Wariant bez `line_no_norm`:

```powershell
python src/train_text_baseline.py `
  ... `
  --use-layout-features `
  --exclude-line-no-feature `
  --output-dir output/layout_experiments/exp_text_svm_layout_no_line_no_seed42
```

### Krok 5 — pełny multi-seed

```powershell
python src/run_multi_seed_experiments.py `
  --with-layout `
  --layout-labels-csv data/labels_synthetic_with_layout.csv `
  --output-root output/layout_experiments `
  --seeds 42 123 456 `
  --exclude-labels 10
```

Uruchamia na każdy seed:

- `text_logreg`, `text_svm` (baseline na layout CSV)
- `text_logreg_layout`, `text_svm_layout`
- `text_logreg_layout_no_line_no`, `text_svm_layout_no_line_no`

**Nie nadpisuje** `output/multi_seed/`.

Szacowany czas: zależny od CPU/GPU; sam trening text to minuty, pełny eksport OCR może trwać dłużej.

### Krok 6 — odczyt wyników

| Plik | Zawartość |
|------|-----------|
| `output/layout_experiments/multi_seed_summary.json` | mean ± std metryk per model_key |
| `output/layout_experiments/multi_seed_summary.csv` | to samo w CSV |
| `output/layout_experiments/multi_seed_per_run.csv` | wyniki per seed × model |
| `output/layout_experiments/per_class_f1.csv` | precision/recall/F1 per klasa per run |
| `output/layout_experiments/exp_{model}_seed{N}/metrics.json` | metryki pojedynczego runu |
| `output/layout_experiments/exp_{model}_seed{N}/confusion_matrix_test.csv` | macierz pomyłek |

### Krok 7 — interpretacja

**Główna metryka:** `test macro F1` (mean ± std z 3 seedów).

Porównania:

| A | B | Pytanie |
|---|---|---------|
| `text_svm` | `text_svm_layout` | Czy bbox poprawia SVM? |
| `text_svm_layout` | `text_svm_layout_no_line_no` | Czy poprawa idzie z bbox czy z kolejności linii? |
| `text_logreg` | `text_logreg_layout` | Czy bbox poprawia LogReg? |
| `text_logreg_layout` | `text_logreg_layout_no_line_no` | j.w. dla LogReg |

Dodatkowo: test accuracy, test weighted F1, per-class F1 (szczególnie `SELLER_*`, `BUYER_*`, `TOTAL_PRICE`, daty, kwoty).

**Próg praktycznej poprawy:** test macro F1 > baseline + **0.01** (mean over seeds).

---

## Tabela wyników

_Po uruchomieniu pełnego multi-seed uzupełnij na podstawie `multi_seed_summary.csv`._

| model_key | test_macro_f1_mean | test_macro_f1_std | test_accuracy_mean | test_weighted_f1_mean |
|-----------|-------------------|-------------------|--------------------|-----------------------|
| _(do uzupełnienia)_ | | | | |

Referencja baseline (stary CSV, `output/multi_seed/`): patrz [`MULTI_SEED_RESULTS.md`](MULTI_SEED_RESULTS.md).

---

## Ograniczenia

- Bbox pochodzi z OCR docTR na **stronie 0** PDF.
- `line_no_norm` zależy od kolejności linii zwracanej przez docTR.
- Dane syntetyczne — layout może być specyficzny dla generatora (`layout_a/b/c`).
- Pełna ocena wymaga re-eksportu wszystkich faktur i multi-seed na pełnym zbiorze.

---

## CNN + layout — rekomendacja

Przejść do **CNN + layout** warto dopiero gdy:

1. modele `text_*_layout` pokażą poprawę test macro F1 ≥ 0.01 vs baseline text, **oraz**
2. ablacja `*_no_line_no` wskaże, że poprawa pochodzi z bbox, nie tylko z `line_no_norm`.

Jeśli layout nie pomaga modelom tekstowym, fusion CNN+layout ma mniejszą szansę na zysk (CNN już widzi część informacji wizualnej z cropu, ale traci bezwzględną pozycję na stronie).

---

## Zmienione / dodane pliki

| Plik | Zmiana |
|------|--------|
| `src/layout_features.py` | **NOWY** — ekstrakcja i walidacja cech |
| `src/export_synthetic_to_labels.py` | flaga `--include-layout-features` |
| `src/train_text_baseline.py` | `--use-layout-features`, `--exclude-line-no-feature` |
| `src/run_multi_seed_experiments.py` | `--with-layout`, agregacja `per_class_f1.csv` |
| `src/validate_layout_csv.py` | **NOWY** — walidacja CSV |
| `tests/test_layout_features.py` | **NOWY** |
| `tests/test_text_layout_training.py` | **NOWY** |
| `tests/conftest.py` | **NOWY** |

Stare komendy bez nowych flag działają bez zmian.
