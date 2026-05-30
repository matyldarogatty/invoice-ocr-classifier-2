# Wyniki eksperymentów multi-seed (100 faktur, 11 klas aktywnych)

Protokół: podział dokumentowy 70/15/15, `--exclude-labels 10` (CURRENCY), seed ∈ {42, 123, 456}.

Uruchomienie:

```powershell
python src/run_multi_seed_experiments.py --seeds 42 123 456 --device cpu --epochs 10
```

Artefakty: `output/multi_seed/` — splity, modele, `multi_seed_per_run.csv`, `multi_seed_summary.json`.

## Podsumowanie (średnia ± odch. std., 3 seedy)

### TF-IDF + Logistic Regression

| Metryka | Val | Test |
|---------|-----|------|
| Accuracy | 0.8422 ± 0.0042 | 0.8381 ± 0.0212 |
| Macro F1 | 0.6024 ± 0.0090 | 0.6083 ± 0.0179 |
| Weighted F1 | 0.8679 ± 0.0052 | 0.8670 ± 0.0196 |

### TF-IDF + LinearSVC

| Metryka | Val | Test |
|---------|-----|------|
| Accuracy | 0.9302 ± 0.0028 | 0.9246 ± 0.0150 |
| Macro F1 | 0.6286 ± 0.0113 | 0.6261 ± 0.0328 |
| Weighted F1 | 0.9197 ± 0.0035 | 0.9168 ± 0.0159 |

### InvoiceCNN + class weights

| Metryka | Val | Test |
|---------|-----|------|
| Accuracy | 0.8705 ± 0.0270 | 0.8771 ± 0.0265 |
| Macro F1 | 0.6569 ± 0.0067 | **0.6758 ± 0.0275** |
| Weighted F1 | 0.8943 ± 0.0158 | 0.8983 ± 0.0165 |

## Wnioski wstępne (multi-seed, N=100 dokumentów)

1. **Macro F1 (test)** — CNN (~**0.676** ± 0.028) **przewyższa** modele tekstowe (~0.608 LogReg, ~0.626 LinearSVC) przy 3 seedach i zregenerowanych splitach na pełnym zbiorze.
2. **Accuracy (test)** — LinearSVC (~0.925) nadal najwyższy, co wynika z dominacji klasy OTHER.
3. Wyniki różnią się od [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) opartego na wcześniejszym, nieaktualnym splicie (10 dokumentów) — **używać `output/multi_seed/` jako źródła prawdy**.

## Porównanie ze starym splitem (10 dokumentów)

Stary `output/splits_70_15_15` (przed regeneracją): 7/2/1 faktury, 956 wierszy — **nie używać w pracy**.

Po regeneracji: 70/15/15 faktur, 4656 wierszy — patrz [DATA_VERIFICATION.md](DATA_VERIFICATION.md).
