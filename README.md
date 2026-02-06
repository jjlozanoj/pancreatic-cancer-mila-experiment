# Pancreatic Cancer MILA Pilot Experiment

This repository contains code, experiment artifacts, and results for a **pilot experiment on pancreatic cancer cases**, designed to evaluate:

- **Image segmentation quality** (qualitative/quantitative inspection of outputs)
- **Attention-based Multiple Instance Learning (MIL)** for **patient-level prediction**
- **Interpretability** via attention heatmaps and extraction/quantification of high-attention morphological patterns

---

## What’s in this repo (high level)

- **End-to-end research scripts**: dataset mapping, dataframe building, MIL attention heatmaps, pattern quantification, and survival analysis.
- **Cohort descriptors**: clinical/metadata tables used to drive the analysis (WSIs are not included).
- **Reproducible outputs**: run folders containing logs, tables, case-level heatmaps/thumbnails, connected components, and crops.
- **Curated figures**: ready-to-use Kaplan–Meier and boxplot figures for train/test splits.

---

## Repository layout

```text
.
├── code/
│   ├── df.py
│   ├── df_att.py
│   ├── inspeccion.py
│   ├── mapear_dataset.py
│   ├── mil_heatmap.py
│   ├── quant_attention_patterns.py
│   └── survival_analysis.py
│
├── data/
│   └── filtered_clinical_data.csv
│
├── experiments/
│   └── (experiment notes / cohort-specific documentation)
│
├── results/
│   ├── inspeccion.txt
│   ├── mapear_dataset.txt
│   └── out_patterns/
│       └── run_<timestamp>/
│           ├── 00_logs/
│           │   └── summary_report.txt
│           ├── 01_tables/
│           │   ├── case_components_and_crops.csv
│           │   ├── case_heatmap_summary.csv
│           │   └── top_patches_meta.csv
│           └── 02_cases/
│               └── <CASE_ID>/
│                   ├── 00_thumb/
│                   │   └── thumb.png
│                   ├── 01_heat/
│                   │   ├── heat_only.png
│                   │   ├── heat_overlay.png
│                   │   └── mask_high.png
│                   ├── 02_components/
│                   │   └── components.json
│                   └── 03_crops/
│                       └── crop_ccXX_LY_LZXXXX.png
│
└── figures/
    ├── km_train.png
    ├── km_test.png
    ├── box_train.png
    └── box_test.png
```

---

## Data and access restrictions

This repo does not ship raw WSIs (e.g., TCGA slides). The data/ folder contains cohort-level tables (e.g., filtered_clinical_data.csv) used to drive the pipeline (case IDs, clinical endpoints, and derived fields). You must obtain and mount the underlying image data separately according to your institution’s and/or TCGA’s data access policies.

---

## Code overview (what each script is for)

The code is organized as a set of research scripts (not a packaged library). Typical roles:

- mapear_dataset.py
Maps/normalizes dataset identifiers and paths; produces a mapping report (see results/mapear_dataset.txt).
- inspeccion.py
Sanity checks / inspections over cohort tables and/or intermediate outputs; produces results/inspeccion.txt.
- df.py, df_att.py
Build or reshape patch-/case-level dataframes used for downstream analysis.
- mil_heatmap.py
Generates attention heatmaps and related visual overlays for interpretability.
- quant_attention_patterns.py
Extracts and summarizes high-attention regions, connected components, and crops; writes the structured outputs under results/out_patterns/run_<timestamp>/.
- survival_analysis.py
Produces survival-related plots and tables (e.g., Kaplan–Meier curves and risk boxplots), with curated outputs stored in figures/.

---

## Output structure (pattern runs)

When running pattern quantification / heatmap extraction, outputs are written under:

- results/out_patterns/run_<timestamp>/00_logs/
Summary logs (summary_report.txt) describing the run.
- results/out_patterns/run_<timestamp>/01_tables/
CSV tables that summarize:
  - case-level component and crop counts
  - heatmap summaries
  - metadata for top patches / regions
-results/out_patterns/run_<timestamp>/02_cases/<CASE_ID>/
Case-level artifacts:
  - 00_thumb/: slide thumbnail(s)
  - 01_heat/: heatmaps (raw + overlay) and binary masks for selected high-attention regions
  - 02_components/: connected component metadata (components.json)
  - 03_crops/: cropped image regions corresponding to detected components

---

## Re-running / adapting to a new environment

Because this is research code, expect to update:

- Filesystem paths (WSIs, patches, embeddings, outputs)
- Cohort table column names (if your CSV schema differs)
- Any hard-coded parameters (patch size, magnification, thresholds)
