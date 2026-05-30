# Weryfikacja lokalnych danych (AI_OCR)

Data weryfikacji: 2026-05-24

## Podsumowanie

| Zasób | Oczekiwane | Stan lokalny | Status |
|-------|------------|--------------|--------|
| `data/labels_synthetic.csv` | 4656 wierszy, 100 faktur | 4656 wierszy, 100 `invoice_id` | OK |
| `data/images_synthetic/*.png` | ~4656 cropów | **4656** plików PNG | OK |
| `synthetic_output/batch_100_v2/pdfs/*.pdf` | 100 PDF | **100** plików PDF | OK |
| Brakujące etykiety / tekst / obrazy | 0 | 0 (audit) | OK |
| Klasa CURRENCY (id 10) | często 0 w eksporcie | **0** wierszy | OK (zgodnie z README) |
| `output/splits_70_15_15/` | 70/15/15 **dokumentów** | **Było nieaktualne (10 docs)** → **zregenerowano** | NAPRAWIONO |

## Audyt zbioru (`audit_dataset.py`)

```
Rows: 4656
Missing labels: 0  |  missing text: 0  |  missing image file: 0
Documents (by invoice_id): 100
Config classes with ZERO examples: id=10 CURRENCY
```

Rozkład klas (skrót): OTHER=3876 (~83%), pozostałe klasy rzadkie (30–100 linii).

## Splity dokumentowe (po regeneracji, seed=42)

```
Documents — train: 70, val: 15, test: 15
Rows      — train: 3230, val: 676, test: 750
```

Plik: `output/splits_70_15_15/split_metadata.json`

**Uwaga:** Wcześniejszy `split_metadata.json` dotyczył tylko **10 dokumentów** (7/2/1, 956 wierszy) — prawdopodobnie utworzony na wcześniejszym, mniejszym CSV. Przed finalnymi eksperymentami należy zawsze weryfikować `n_documents_train + val + test = 100`.

## Wnioski dla pracy magisterskiej

1. Pełny zbiór syntetyczny (100 faktur, 4656 linii, cropy, PDF) jest **dostępny lokalnie** i gotowy do treningu.
2. Splity należy **regenerować** po każdej zmianie `labels_synthetic.csv`.
3. Metryki w `docs/RESULTS_SUMMARY.md` mogły pochodzić ze starego splitu (10 docs) lub nowego (100 docs) — po multi-seed run należy używać `output/multi_seed_summary.*`.
