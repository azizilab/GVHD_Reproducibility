# MLR Pipelines — TCR Repertoire Analysis for GVHD

Reproducibility pipeline for the Mixed Lymphocyte Reaction (MLR) T-cell receptor repertoire analysis supporting the manuscript:

This pipeline generates all manuscript figures from Figures 1–2 (panels e–h) and Supplementary Figures 1–4, 8–9 from raw immunoSeq data.

---

## Directory Structure

```
MLR_Pipelines/
├── run_all.R                          # Entry point: runs all notebooks in order
├── config/
│   └── config.R                      # Paths, parameters, and package loading
├── functions/
│   └── mlr_functions.R               # All analysis and plotting functions (30 functions)
├── notebooks/
│   ├── 00_data_loading_corrected.Rmd # Data loading, clone identification, R098 fix
│   ├── 01_patient_figures_and_diversity.Rmd  # Per-patient QC and diversity
│   ├── 02_clone_analysis_corrected_fig1C.Rmd # Fig 1c, Supp 4a/c–f, Supp 9a–c
│   ├── 03_grade_ptcy_analysis_corrected.Rmd  # Fig 1e/1f/1g, Supp 4b/g–j, Supp 8b–f
│   ├── 04_cumulative_figures_corrected.Rmd   # Fig 1d, Supp 2a/2b, Supp 8c/e
│   └── 05_figure3_annotation_variants.Rmd   # Fig 2f/2g/2h (p-value and asterisk variants)
├── MLR_rcode_work_for_all_sample_combined_final_ptcy_github.Rmd       # PTCy alloreactive clone dynamics
├── MLR_rcode_work_for_all_sample_combined_final_tissue_analysis_github.Rmd  # Tissue/anatomical site analysis
├── Data/
│   └── Data_Instructions.txt         # Instructions for downloading raw data
└── Basis_Hydrophobicity_CDR3Length_Figures.Rmd  # Supplementary CDR3 analysis
```

> **Note on original notebooks:** `MLR_rcode_work_for_all_sample_combined_github_final_essential.Rmd` is the original monolithic analysis notebook; the `notebooks/00–05` pipeline was derived from it with corrected paths, the R098 fix, and modular organization. The PTCy and tissue notebooks are companion analyses for PTCy alloreactive clone dynamics (Fig 2/Figure3 outputs) and exploratory tissue/anatomical site clone tracking, respectively.

---

## Requirements

### Conda Environment

All notebooks must be run inside the `MLR` conda environment. To create it:

```bash
conda env create -f environment.yml
conda activate MLR
```

Key packages: `R >= 4.0`, `cdr3tools`, `tidyverse`, `ggplot2`, `ggpubr`, `rstatix`, `ggbeeswarm`, `cowplot`, `KneeArrower`, `readxl`, `patchwork`.

### Data

Download the raw immunoSeq data files and place them in the `Data/` folder following the instructions in `Data/Data_Instructions.txt`. The pipeline also requires:

- `TCR_Project.xlsx` — patient metadata (place in the pipeline root)
- Adaptive Biotechnologies immunoSeq `.tsv` files organized per patient in `Data/`

---

## Running the Pipeline

From the `MLR_Pipelines/` directory:

```bash
conda run -n MLR Rscript run_all.R
```

This runs all six notebooks in sequence. **Order matters** — each notebook reads `.rds` files written by earlier ones:

| Order | Notebook | Writes |
|-------|----------|--------|
| 00 | `00_data_loading_corrected.Rmd` | `data/processed_corrected/` (expanded clone RDS files) |
| 01 | `01_patient_figures_and_diversity.Rmd` | 
| 02 | `02_clone_analysis_corrected_fig1C.Rmd` | 
| 03 | `03_grade_ptcy_analysis_corrected.Rmd` | 
| 04 | `04_cumulative_figures_corrected.Rmd` | 
| 05 | `05_figure3_annotation_variants.Rmd` | 

To render a single notebook:

```bash
conda run -n MLR Rscript -e \
  "rmarkdown::render('notebooks/03_grade_ptcy_analysis_corrected.Rmd', output_dir='notebooks')"
```

---

## Output

All figures are written to `Main_fig/`:

```
Main_fig/
├── Figure1/        # Fig 1b–g (main paper panels)
├── Figure1ex/      # Supp Figs 1–4, 8 (supplementary panels)
├── Figure3/        # Fig 2f–h (PTCy/Shannon panels)
└── Figure3ex/      # Supp Fig 9 (alloreactive clone panels)
```

Each figure is saved as `.png` and `.pdf` at 600 dpi. Every panel also has:
- A `_sample_size` companion bar chart (N per group per time bin)
- A `_pvalue` companion showing exact p-values (where applicable)

---

## Manuscript Figure Mapping

| Manuscript panel | File(s) in `Main_fig/` | Notebook |
|---|---|---|
| Fig 1b | `Figure1/AlloFrequnceyDotPlot` | 01 |
| Fig 1c | `Figure1/C_365_signif_hochberg` | 02 |
| Fig 1d | `Figure1/severe_cumulative`, `mild_cumuluative`, `no_cumuluative` | 04 |
| Fig 1e/1f | `Figure1/fig_EF_pval` | 03 |
| Fig 1g | `Figure1/All_by_grade_fig` | 03 |
| Fig 2f | `Figure3/CombinedPTCy_365_consistent` | 05 |
| Fig 2g | `Figure3/all_All_byPTCy_pvalue` | 05 |
| Fig 2h | `Figure3/ByPTCy_Shannon_GVHD_GRADE_1_noSlopeFilter` | 05 |
| Supp Fig 1a–d | `Figure1ex/*DotPlots*` | 01–02 |
| Supp Fig 2a–b | `Figure1ex/*cumulative*` | 04 |
| Supp Fig 3 | `Figure1/C_nonallo_365_hochberg` | 02 |
| Supp Fig 4a–j | `Figure1ex/freq_*`, `Grade_*`, `BothPTCy_*` | 02–03 |
| Supp Fig 8b–f | `Figure3/freq_Both_vio_ex`, `BothPTCy_*`, `All_by_ptcy_fig`, `all_All_nonallo_byPTCy` | 03 |
| Supp Fig 9a–d | `Figure3ex/freq_Both_vio_corrected`, `freq_CD4/CD8_vio_*`, `CD4/CD8_byPTCy*` | 02–03 |

> **Note:** Fig 2 panels a–d are from the DecompTCR pipeline (separate repository).  
> Supp Fig 8a has no corresponding code in this pipeline.

---

---

## Functions Reference

See `functions/README.md` for documentation of all 30 functions organized into five categories: data processing, counting/metrics, analysis, visualization, and utilities.

---

