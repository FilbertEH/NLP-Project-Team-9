# NLP-Project-Team-9

## Run
1. Download the raw GDELT Parquet Archive (gdelt_id_\*.parquet) files and place them into the data/raw/ directory
1a. Here's the link to the GDrive: https://drive.google.com/drive/folders/1pUpsC_wfg-NClHkabtEsptIOAerlE_23

2. Verify Bank Indonesia JISDOR Exchange Rate Data
The cleaned exchange rate series is provided at data/info_kurs.csv and data/jisdor_cleaned.csv. If running from raw Bank Indonesia exports, ensure the spreadsheet is located at data/raw/jisdor/jisdor_raw.xlsx as defined in config.yaml

## End-to-End Execution Guide
Follow these sequential steps to reproduce the entire pipeline from scratch.

### Step 1: Pre-filtering and Queue Construction
Normalises article URLs, eliminates duplicates, filters out non-economic editorial desks (e.g. lifestyle, celebrity, sports), matches macroeconomic and geopolitical GDELT theme prefixes (ECON_, EPU_, TAX_FNCACT, ARMEDCONFLICT, TRADE, CRISISLEX), and converts UTC timestamps to Western Indonesia Time (WIB)

```python -m src.selection.build_queue```

### Step 2: Full-Body Web Scraping
Executes concurrent polite web scrapers equipped with exponential backoff and Internet Archive Wayback Machine snapshot fallbacks to retrieve complete article bodies

```python -m src.scrapers.run_all```

### Step 3: Text Sanitisation and Boilerplate Stripping
Unescapes HTML entities, removes cross-site syndicated duplicate titles, and strips portal-specific editorial disclaimers and author bylines

```python -m src.preprocessing.clean_text```

### Step 4: Load and Format JISDOR Rate Target Series
Calculates daily percentage returns r_t, log returns, directional binary movement flags, and calendar gap intervals

```python -m src.fx.load_jisdor```

### Step 5: Causal Forward Temporal Alignment
Maps each article forward to the nearest Bank Indonesia JISDOR fixing published at or after its capture timestamp, applying the strict 10.00 am WIB cutoff

```python -m src.alignment.align_news```

### Step 6: Daily News Bundle and Dataset Aggregation
Merges the daily JISDOR series with aggregate news metrics (daily article volumes, outlet distributions, average tone, and concatenated daily headlines)

```python -m src.alignment.build_dataset```

Output: data/processed/final_dataset.csv

### Step 7: Classical NLP Feature Extraction
Extracts domain-adapted Loughran-McDonald financial sentiment scores, macroeconomic uncertainty indices, institutional and geopolitical entity frequencies, and 15 orthogonal Latent Semantic Analysis (LSA) topics

```python -m src.features.extract_features```

Output: data/processed/dataset_with_nlp_features.csv

### Step 8: Strict Chronological Data Partitioning
Applies annual partition boundaries to generate anti-leakage training, validation, and test datasets stored in data/:

```python -m src.data.split_data```

Outputs: data/train.csv, data/val.csv, data/test.csv

### Step 9: Baseline and Combined Model Training
Trains the naive benchmark, pure time-series baselines (autoregressive lags and rolling volatilities), and multi-modal combined models:

```python -m src.models.train_models```

Output: Metrics saved to data/processed/model_benchmark_results.json

### Step 10: Run the Diagnostic Evaluation Suite
Formats and prints complete classification diagnostics, confusion matrices, and probability calibration scores:

```python -m src.models.evaluate```

