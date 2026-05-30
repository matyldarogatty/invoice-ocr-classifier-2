# AI_OCR — klasyfikacja semantyczna linii faktury

## 1. Cel projektu

Projekt dotyczy **klasyfikacji pojedynczych linii tekstu** wyodrębnionych z faktur PDF (typowo polskich układów): dla każdej linii OCR system przypisuje **klasę semantyczną** z ustalonej taksonomii (np. dane sprzedawcy, NIP-y, kwoty, daty, klasa „reszta” — **OTHER**).

To **nie jest** gotowy system księgowy ani narzędzie do **pełnej ekstrakcji strukturalnej faktury do JSON** z walidacją NIP, kwot i dat. Nie wybiera automatycznie „jednej właściwej” wartości pola z wielu linii ani nie integruje się z KSeF.

Celem badawczym — zwłaszcza w pracy magisterskiej — jest **porównanie kilku metod uczenia maszynowego** na tych samych (lub porównywalnych) podziałach danych: sieć konwolucyjna na **obrazach linii** oraz modele **tekstowe TF–IDF + klasyfikator liniowy** na tekście z OCR.

---

## 2. Co robi program

**Przepływ danych (uproszczony):**

```text
PDF → OCR (docTR, wstępnie wytrenowany) → linie (tekst + geometria)
   → (opcjonalnie) cropy obrazów linii
   → klasyfikator ML → etykieta semantyczna na linię
```

**Tryb badawczy:** budowa zbioru (syntetycznego lub pomocniczo manualnego), **audyt**, **podział dokumentowy** train/val/test, **trening** modeli (CNN oraz baseline tekstowy), **metryki** (`metrics.json`, raporty, macierze pomyłek), **agregacja porównawcza** (`compare_experiments.py`).

**Tryb demonstracyjny:** uruchomienie **`run_invoice_inference.py`** na **jednym pliku PDF**: OCR linii, klasyfikacja wybraną metodą (**CNN** lub **tekst**), zapis **`predictions.csv`** oraz cropów do porównań wizualnych — **bez** liczenia accuracy/F1 na tej fakturze (brak referencyjnych etykiet dla Twojego PDF, o ile nie dodasz ich ręcznie).

Szczegółowy protokół eksperymentów i wstępne tabele wyników: **[EXPERIMENTS.md](EXPERIMENTS.md)**, **[docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)**. Stan modułów: **[docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md)** (część opisu inferencji może być starsza niż aktualny skrypt — za źródło prawdy dla CLI przyjmij ten README i kod **`run_invoice_inference.py`**).

---

## 3. Porównywane metody

| Metoda | Modality | Opis |
|--------|-----------|------|
| **CNN (`InvoiceCNN`)** | Obraz | Mała sieć konwolucyjna na **szarym cropie** linii (**128×128**), [`src/train.py`](src/train.py). |
| **TF–IDF + Logistic Regression** | Tekst OCR | Wektor n-gramów + regresja logistyczna, [`src/train_text_baseline.py`](src/train_text_baseline.py) (`--model logistic_regression`). |
| **TF–IDF + LinearSVC** | Tekst OCR | Jak wyżej z **`LinearSVC`** (`--model linear_svc`). |

**Obraz vs tekst:** CNN widzi układ pikseli (czcionka, marginesy, „wygląd” linii), więc może być **czulszy na layout i szum skanu**, ale nie musi polegać na poprawnym rozpoznaniu każdego słowa. Modele tekstowe bazują wyłącznie na **łańcuchu znaków z OCR** — przy dobrym OCR na fakturach strukturalnych mogą **silnie wykorzystywać słowa kluczowe** („NIP”, „Razem”, format daty). Żadna z tych metod nie jest tu „hybrydą” obraz+tekst w jednym modelu.

Model OCR **docTR** jest używany **do ekstrakcji linii**, nie jest dalej trenowany w tym repozytorium.

---

## 4. Struktura projektu

| Katalog / plik | Rola |
|----------------|------|
| **[synthetic_invoice_generator/](synthetic_invoice_generator/)** | Generator syntetycznych faktur PDF + JSON z podpowiedziami semantycznymi; wyjście m.in. **`manifest.jsonl`**. Ma własny **`requirements.txt`**. |
| **[src/](src/)** | Cała logika: konfiguracja (`config.py`), eksport etykiet, splity, trening, metryki, inferencja jednej faktury. |
| **[data/](data/)** | Dane robocze: **`images_synthetic/`**, **`labels_synthetic.csv`**, opcjonalnie **`raw/`** (PDF-y do demo), **`labels.csv`** / **`images/`** (ścieżka manualna). |
| **[output/](output/)** | Artefakty wyjściowe: audyty, **splity**, katalogi **`exp_*`** z modelami i metrykami, **`predictions/`** z wynikami inferencji. Często objęte `.gitignore`. |
| **[models/](models/)** | Miejsce na **legacy** checkpointy (np. starszy CNN); **nie** należy ich mylić z **`output/exp_*/model.pth`** z eksperymentów 11-klasowych. |
| **[docs/](docs/)** | Skrótowe dokumenty techniczne i podsumowanie wstępnych metryk. |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | Pełniejszy opis pipeline’u eksperymentów i parametrów. |
| **[PROJECT_CURRENT_STATE.md](PROJECT_CURRENT_STATE.md)** | Snapshot kontekstu projektu (ograniczenia, liczba klas w eksperymentach). |

---

## 5. Dane

- **Dane syntetyczne:** generowane przez **`python -m synthetic_invoice_generator`** → dla każdej „faktury” powstaje PDF, plik JSON z m.in. **`classification_hints`** oraz wpis w **`manifest.jsonl`**.
- **`data/labels_synthetic.csv`:** zbiór **linii** po eksporcie ([`export_synthetic_to_labels.py`](src/export_synthetic_to_labels.py)): ścieżka cropu, tekst OCR, etykieta numeryczna, **`invoice_id`**, itd.
- **`data/images_synthetic/`:** obrazy (PNG) cropów wskazywane przez kolumnę **`filename`** w CSV.
- **`data/labels_synthetic_review.csv`:** zwykle pełniejszy zapis pomocniczy do przeglądu dopasowań (generowany przy eksporcie).
- **`data/labels_synthetic_summary.json`:** opcjonalne podsumowanie liczbowe eksportu (np. **`--summary-json-path`** w eksporcie).
- **`data/raw/`:** naturalne miejsce na **własne PDF-y** do inferencji demonstracyjnej (`run_invoice_inference.py`); katalog może być pusty w repozytorium.

**Ostrzeżenie — dwie ścieżki danych:**

- **`data/labels.csv`** oraz **`data/images/`** często wiążą się ze **ścieżką manualną / legacy** (np. [`OCR.py`](src/OCR.py) — nadpisuje CSV; zwykle **brak kolumny dokumentowej** wymaganej do dokumentowego splitu).
- **Główny pipeline eksperymentalny** opisany w tym README powinien opierać się na **`labels_synthetic.csv`** + **`images_synthetic/`** oraz splitalch utworzonych z tego CSV.

---

## 6. Klasy semantyczne

Taksonomia (identyfikatory numeryczne → nazwy) jest **jednoźródłowo** zdefiniowana w **[`src/config.py`](src/config.py)** — **12 klas**:

| ID | Nazwa w kodzie |
|----|----------------|
| 0 | SELLER_NAME |
| 1 | SELLER_NIP |
| 2 | BUYER_NAME |
| 3 | BUYER_NIP |
| 4 | TOTAL_PRICE |
| 5 | INVOICE_NUMBER |
| 6 | INVOICE_DATE |
| 7 | SALE_DATE |
| 8 | NET_AMOUNT |
| 9 | VAT_AMOUNT |
| 10 | CURRENCY |
| 11 | OTHER |

**OTHER:** klasa „wszystko inne” — linie, których nie udało się jednoznacznie przypisać do konkretnego pola (albo są nietypowe). W eksporcie syntetycznym **OTHER często dominuje** (wysoki udział linii tabeli, opisów, nagłówków bez dopasowania reguły).

**Wykluczenie CURRENCY (`--exclude-labels 10`) w eksperymentach „no_currency”:** w wielu partiach syntetycznych **nie ma osobnych linii z etykietą CURRENCY** (waluta bywa zlepiona z linią kwoty). Eksperymenty główne **usuwają te wiersze tylko w pamięci** podczas treningu; **`config.LABELS`** i ID **10** pozostają w kodzie. Checkpoint ma wtedy **11 wyjść** + **`label_mapping.json`** — **nie** mylić z pełnym modelem 12-klasowym ani z legacy checkpointem 7-klasowym.

---

## 7. Instalacja

**Wymagania:** Python z obsługą venv; dostęp do internetu przy pierwszym `pip install`.

**Windows (PowerShell), z katalogu głównego repozytorium:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r synthetic_invoice_generator/requirements.txt
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r synthetic_invoice_generator/requirements.txt
```

**[`requirements.txt`](requirements.txt)** (katalog główny) zestawia biblioteki do **PyTorch**, **scikit-learn**, **python-doctr**, **pandas**, itd. — potrzebne do treningu, eksportu i inferencji. Oddzielny plik **[`synthetic_invoice_generator/requirements.txt`](synthetic_invoice_generator/requirements.txt)** dopina zależności **generatora PDF** (np. WeasyPrint). Bez drugiego kroku generator może nie działać.

---

## 8. Generowanie danych syntetycznych

Komenda modułu (CLI zweryfikowane przez `python -m synthetic_invoice_generator --help`):

```powershell
python -m synthetic_invoice_generator --count 100 --seed 42 --out-dir synthetic_output/batch_100
```

**Co powstaje** (względem `--out-dir`):

- Katalog z **PDF-ami** faktur (np. podfolder **`pdfs/`** — dokładna struktura zależy od wersji generatora; warto sprawdzić zawartość `out-dir` po uruchomieniu).
- **JSON-y** z ground truth (np. **`json/`**).
- **`manifest.jsonl`** — lista przetworzonych dokumentów (wejście do eksportu OCR).

Przy ponownym użyciu niepustego katalogu może być potrzebne **`--overwrite`** (patrz help modułu).

---

## 9. Eksport OCR i etykiet

Skrypt **[`src/export_synthetic_to_labels.py`](src/export_synthetic_to_labels.py)**:

- **Wejście:** ścieżka do **`manifest.jsonl`**, katalog **PDF**, katalog **JSON** zgodny z generatorem.
- **Działanie:** docTR na PDF → linie → cropy; dopasowanie linii OCR do **`classification_hints`** z JSON przez **[`match_utils`](src/match_utils.py)** (reguły; przy niejednoznaczności często **OTHER**).
- **Wyjście:** katalog **`images-dir`** z PNG oraz **`csv-path`** (główny CSV etykiet), **`review-csv-path`** (CSV przeglądowy). Opcjonalnie **`--summary-json-path`** dla statystyk.

Domyślnie skrypt **odmawia** nadpisania **`data/labels.csv`** i **`data/images/`** — używaj np. **`data/images_synthetic`** i **`data/labels_synthetic.csv`**.

Przykład (dostosuj ścieżki do swojego `--out-dir` generatora):

```powershell
python src/export_synthetic_to_labels.py --manifest synthetic_output/batch_100/manifest.jsonl --pdf-dir synthetic_output/batch_100/pdfs --json-dir synthetic_output/batch_100/json --images-dir data/images_synthetic --csv-path data/labels_synthetic.csv --review-csv-path data/labels_synthetic_review.csv --summary-json-path data/labels_synthetic_summary.json
```

---

## 10. Audyt danych

**[`src/audit_dataset.py`](src/audit_dataset.py)** — odczyt **tylko** CSV + katalogu obrazów; generuje raporty jakości (kompletność etykiet, obecność plików, rozkład klas, opcjonalnie rozklad dokumentów).

Przykład:

```powershell
python src/audit_dataset.py --labels-csv data/labels_synthetic.csv --images-dir data/images_synthetic --output-dir output/audit_synthetic_100 --force
```

**`--force`** pozwala nadpisać poprzednie pliki w `--output-dir`. Typowe wyjścia: **`dataset_audit_summary.json`**, **`class_distribution.csv`**.

---

## 11. Tworzenie podziału train / val / test

**[`src/create_splits.py`](src/create_splits.py)** + **[`splitting.py`](src/splitting.py)**:

- **Podział dokumentowy:** domyślnie po **`invoice_id`** (alternatywnie **`source_pdf`**, jeśli podasz **`--group-column`** lub brak `invoice_id`).
- **Dlaczego nie losujemy pojedynczych linii:** linie z jednej faktury mają wspólny layout i styl — losowy podział linii **przenosiłby informację między train a test** (zawyżenie metryk).
- Domyślne proporcje: **70% / 15% / 15%** dokumentów; **`--seed`** (domyślnie **42**).
- **Wyjście:** `train.csv`, `val.csv`, `test.csv`, **`split_metadata.json`** (counts, rozkłady klas).

```powershell
python src/create_splits.py --labels-csv data/labels_synthetic.csv --output-dir output/splits_70_15_15 --seed 42
```

**Ważne ostrzeżenie:** jeśli **`labels_synthetic.csv`** został **powiększony po wygenerowaniu splitów**, stare pliki w **`output/splits_70_15_15/`** mogą obejmować **tylko część dokumentów** — wartość **`split_metadata.json`** i liczby wierszy **nie muszą** pokrywać się z aktualnym pełnym CSV. Do **finalnego eksperymentu** należy **ponownie uruchomić `create_splits.py`** na aktualnym zbiorze i zapisać np. nowy katalog:

```powershell
python src/create_splits.py --labels-csv data/labels_synthetic.csv --output-dir output/splits_100_70_15_15 --seed 42
```

(nazwa katalogu jest przykładowa).

---

## 12. Trening CNN

**[`src/train.py`](src/train.py)** — tryb z CSV splitów:

- **Wejście:** `--train-csv`, `--val-csv`, `--test-csv`, `--images-dir`, `--output-dir`, oraz opcje: `--epochs`, `--batch-size`, `--device`, `--seed`, **`--exclude-labels`**, **`--use-class-weights`**, opcjonalnie downsampling OTHER na train (`--downsample-label`, `--downsample-ratio`).
- **Wyjście (w `--output-dir`):** **`model.pth`**, **`metrics.json`**, **`config.json`**, **`label_mapping.json`** (przy wykluczeniach), **`classification_report_val.json`**, **`classification_report_test.json`**, **`confusion_matrix_*.csv`**.

Przykład — CNN bez klasy CURRENCY, z wagami klas:

```powershell
python src/train.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --images-dir data/images_synthetic --output-dir output/exp_cnn_no_currency_weighted_run1 --epochs 10 --batch-size 32 --device cpu --seed 42 --exclude-labels 10 --use-class-weights
```

Istnieje też **`--legacy-80-20`** (stary podział na `data/labels.csv`) — **poza** głównym protokołem porównawczym.

---

## 13. Trening modeli tekstowych

**[`src/train_text_baseline.py`](src/train_text_baseline.py)**:

- **Wejście:** te same trzy CSV splitów; wymagane kolumny **`text`** i **`label`**.
- **Modele:** `--model logistic_regression` (domyślnie) lub **`linear_svc`**.
- **Wyjście:** **`model.joblib`** (pipeline TF–IDF + klasyfikator), **`metrics.json`**, **`config.json`**, **`label_mapping.json`**, raporty i macierze — analogicznie do CNN.

**Logistic Regression:**

```powershell
python src/train_text_baseline.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --output-dir output/exp_text_logreg_no_currency_run1 --exclude-labels 10
```

**LinearSVC:**

```powershell
python src/train_text_baseline.py --train-csv output/splits_70_15_15/train.csv --val-csv output/splits_70_15_15/val.csv --test-csv output/splits_70_15_15/test.csv --output-dir output/exp_text_svm_no_currency_run1 --exclude-labels 10 --model linear_svc
```

---

## 14. Porównanie eksperymentów

**[`src/compare_experiments.py`](src/compare_experiments.py)** zbiera **`metrics.json`** z podanych katalogów eksperymentów i zapisuje jeden **CSV** porównawczy.

Kluczowe kolumny wyjściowe m.in.: **`test_macro_f1`**, **`test_weighted_f1`**, **`test_accuracy`**, oraz analogicznie dla **val**.

Przykład dla czterech referencyjnych uruchomień:

```powershell
python src/compare_experiments.py --experiment output/exp_cnn_no_currency_weighted_run1 --experiment output/exp_cnn_no_currency_weighted_downsampled_run1 --experiment output/exp_text_logreg_no_currency_run1 --experiment output/exp_text_svm_no_currency_run1 --output-csv output/experiment_comparison_final_run1.csv
```

**Interpretacja:** przy dominacji **OTHER** najbardziej informacyjna jest zwykle **`macro F1`** (średnia po klasach z równymi wagami). **`accuracy`** może być wysokie przy częstym przewidywaniu klasy dominującej — **nie** powinno być jedynej miary sukcesu.

**Uwaga:** opcja **`--experiments-dir output`** dodaje **wszystkie** podkatalogi ze `metrics.json` (np. uruchomienia „smoke”) — do pracy magisterskiej lepiej wskazać jawnie **`--experiment`** dla wybranych katalogów.

Na dysku deweloperskim może istnieć już plik **[`output/experiment_comparison_final_run1.csv`](output/experiment_comparison_final_run1.csv)** — należy sprawdzić w aktualnym stanie repozytorium.

---

## 15. Inferencja jednej faktury

**[`src/run_invoice_inference.py`](src/run_invoice_inference.py)** — wymaga argumentów:

| Argument | Znaczenie |
|----------|-----------|
| `pdf_path` | Ścieżka do PDF (pozycyjnie). |
| `--method` | `cnn` lub `text`. |
| `--model-path` | CNN: **`model.pth`** z katalogu eksperymentu; tekst: **`model.joblib`**. |
| `--label-mapping` | **`label_mapping.json`** z **tego samego** eksperymentu (format z **`training_to_original`**). |
| `--output-dir` | Katalog na **`predictions.csv`** i podkatalog **`crops/`**. |

**CNN:**

```powershell
python src/run_invoice_inference.py data/raw/mojafaktura.pdf --method cnn --model-path output/exp_cnn_no_currency_weighted_run1/model.pth --label-mapping output/exp_cnn_no_currency_weighted_run1/label_mapping.json --output-dir output/predictions/mojafaktura_cnn
```

**Tekst — LinearSVC:**

```powershell
python src/run_invoice_inference.py data/raw/mojafaktura.pdf --method text --model-path output/exp_text_svm_no_currency_run1/model.joblib --label-mapping output/exp_text_svm_no_currency_run1/label_mapping.json --output-dir output/predictions/mojafaktura_text_svm
```

**Tekst — Logistic Regression:**

```powershell
python src/run_invoice_inference.py data/raw/mojafaktura.pdf --method text --model-path output/exp_text_logreg_no_currency_run1/model.joblib --label-mapping output/exp_text_logreg_no_currency_run1/label_mapping.json --output-dir output/predictions/mojafaktura_text_logreg
```

**Wyniki:**

- **`{output-dir}/predictions.csv`** — jedna linia na zaklasyfikowaną linię OCR (po filtrze minimalnego bbox).
- **`{output-dir}/crops/`** — PNG cropów (do porównań między metodami).

**Kolumny CSV (minimalny zestaw):** `line_no`, `text`, `predicted_training_id`, `predicted_original_id`, `predicted_label`, `method`, `crop_path`.

To nadal jest **klasyfikacja linii**, a nie kompletny JSON faktury ani walidacja księgowa.

---

## 16. Ważne artefakty wynikowe

| Plik | Krótki opis |
|------|-------------|
| **`config.json`** | Parametry uruchomienia: ścieżki CSV, seed, wykluczenia klas, device, itp. |
| **`metrics.json`** | Metryki zbiorcze (**val** / **test**): accuracy, macro F1, weighted F1, **`per_class`**. |
| **`label_mapping.json`** | Mapowanie indeksów treningowych ↔ oryginalnych ID po **`--exclude-labels`**; **wymagane** przy inferencji dla modeli 11-klasowych. |
| **`classification_report_test.json`** | Precision / recall / F1 / support per klasa (test). |
| **`confusion_matrix_test.csv`** | Macierz pomyłek (test): wiersze „prawda”, kolumny „predykcja”. |
| **`model.pth`** | Wagi PyTorch **`InvoiceCNN`** (nie architektura + nie zawsze 12 wyjść). |
| **`model.joblib`** | Pipeline sklearn (TF–IDF + klasyfikator). |
| **`experiment_comparison_*.csv`** | Zestawienie metryk z wielu eksperymentów. |
| **`predictions.csv`** | Wynik inferencji na jednym PDF (demo). |

---

## 17. Znane ograniczenia

- Dane **syntetyczne** mogą nie odzwierciedlać skanów, szumu drukarki ani rzeczywistego rozłożenia pól w PDF zewnętrznych systemów.
- **OTHER** często dominuje — metryki muszą uwzględniać **macro F1** i analizę per klasa.
- **Stare splity** mogą pokrywać tylko część aktualnego **`labels_synthetic.csv`** — przed finalnymi wnioskami wygeneruj **świeże splity**.
- **Jeden seed** (np. 42) i małe zbiory testowe dają **niestabilne** oszacowanie **`test_*`** — rozważ więcej dokumentów i wiele seedów.
- **`models/invoice_cnn.pth`** bywa **legacy** (np. **7 klas**) i **nie** odpowiada checkpointom **`output/exp_cnn_*/model.pth`** z eksperymentów **11-klasowych** — nie mieszać w opisie pracy.
- **Inferencja jednej faktury** nie dostarcza accuracy/F1 **bez ręcznie przygotowanych etykiet referencyjnych** dla tego PDF.
- Projekt **nie** buduje kompletnego JSON-a wartości pól faktury — tylko **etykiety linii**.

---

## 18. Rekomendowany workflow (praca magisterska)

1. Wygeneruj partię syntetycznych faktur (`synthetic_invoice_generator`).
2. Wyeksportuj OCR + etykiety (`export_synthetic_to_labels.py`).
3. Uruchom audyt (`audit_dataset.py`).
4. **Utwórz świeże splity** na aktualnym CSV (`create_splits.py`).
5. Wytrenuj CNN (`train.py`).
6. Wytrenuj modele tekstowe (`train_text_baseline.py`).
7. Porównaj metryki (`compare_experiments.py`).
8. Przeanalizuj **`classification_report_*`** i **`confusion_matrix_*`** dla wybranych modeli.
9. Zademonstruj działanie na jednym PDF (`run_invoice_inference.py`).

---

## 19. Jak interpretować wyniki

- **Wybór „najlepszego” modelu** w porównaniu tabelarycznym opieraj głównie na **`test_macro_f1`** (przy tych samych splitach i taksonomii aktywnych klas).
- **`test_weighted_f1`** i **`test_accuracy`** są **pomocnicze** (bardziej „ważą” klasy liczne).
- **`per_class` / classification report** pokazuje, które pola semantyczne są rozpoznawane, a które konfundowane.
- **Macierz pomyłek** ujawnia systematyczne zamiany klas (np. podobne napisy między typami kwot).

Szczegóły liczb ze **jednego** snapshotu (nie są finałem bez replikacji): **[docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)**.

---

## 20. Podsumowanie

Projekt **AI_OCR** realizuje **klasyfikację semantyczną linii** z faktury PDF przy użyciu OCR **docTR**, na potrzeby **porównania metod ML** (CNN na cropach vs TF–IDF + modele liniowe na tekście). Taksonomia jest stała w **`config.py`** (12 klas w konfiguracji; typowe eksperymenty **bez CURRENCY** mają **11 aktywnych wyjść** i **`label_mapping.json`**). Pipeline badawczy obejmuje generację syntetycznych danych, eksport etykiet, **podział dokumentowy**, trening, metryki i zestawienie **`compare_experiments.py`**. Tryb demonstracyjny zapisuje **`predictions.csv`** i cropy dla pojedynczej faktury. Projekt **świadomie nie realizuje** pełnej ekstrakcji strukturalnej ani walidacji księgowej. Przed silnymi wnioskami warto **odświeżyć splity** na pełnym zbiorze, rozważyć **większą liczbę dokumentów i seedów**, oraz jasno rozróżniać checkpointy **`output/exp_*`** od **legacy** w **`models/`**.

---

## Co warto commitować

Duże zbiory (wszystkie PDF-y, dziesiąki tysięcy PNG, całe `output/exp_*`) często są **wykluczone przez `.gitignore`**. Do pracy magisterskiej zapisuj **hash commita**, dokładne komendy CLI i ścieżki artefaktów użyte do tabel.
