## pancreatic-cancer-mila-experiment
This repository contains the code, experimental setup, and results for a pilot experiment conducted on pancreatic cancer cases. The goal of this experiment is to evaluate image segmentation quality and attention-based multiple instance learning for patient-level prediction.

# Repository Structure
The repository is organized to clearly separate code, data descriptors,
experimental outputs, and visual results. This structure is designed to be
easily extensible to additional cohorts in future experiments.

**`code/`** Contains the scripts used to run the experiment end-to-end, including
  training, inference, and visualization utilities.  
  The code corresponds to the implementation used in the TCGA 40-case experiment.
  
**`data/`**  
  Contains cohort-level metadata and descriptors (e.g., patient-level CSV files).
  Raw TCGA data is not included due to data access restrictions.
  Each cohort is intended to be stored as a separate CSV file.
  
**`experiments/`**  
  Contains notes and documentation specific to each experiment or cohort.
  The current release includes only the TCGA 40-case pilot experiment.
  
**`results/`**  
  Stores full experimental outputs, including quantitative metrics,
  attention maps, and segmentation results generated during inference.
  
**`figures/`**  
  Contains a curated subset of qualitative examples (e.g., attention and
  segmentation visualizations) selected for inspection and presentation.
  

The directory structure is cohort-agnostic by design and allows new datasets
to be incorporated by adding new metadata files and experiment folders,
without modifying the existing codebase.
