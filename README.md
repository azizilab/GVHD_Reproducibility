# GVHD_Reproducibility
GVHD Data Analysis

Overview

This repository contains the code and data analysis pipeline for our study on acute Graft-versus-Host Disease (GVHD) following allogeneic hematopoietic cell transplantation (allo-HCT) in collaboration with Reshef Lab. Acute GVHD is a potentially fatal complication where donor T cells attack recipient tissues. Our study aims to identify reliable biomarkers and uncover the mechanisms of T cell-mediated damage using a combination of in vitro assays, high-throughput sequencing, and spatial transcriptomics.

Study Design

To understand the interaction between donor T cells and host tissues:

We performed mixed lymphocyte reactions (MLRs) combined with high-throughput TCR sequencing to identify alloreactive T cell fingerprints across 20 patients, with 5–12 timepoints per patient. A subset of patients received Post-Transplant Cyclophosphamide (PTCy).
We profiled immune cells in gut biopsies from 14 GVHD patients and two normal donors using paired single-cell transcriptomics and TCR sequencing.
We further examined spatial transcriptomics data to elucidate the tissue architecture and cell-cell interactions underlying GvHD.

Data Access

- TCR raw data: https://drive.google.com/drive/folders/1bum2Z2LbJ6CuspDRHoZm_UG5jxtA4tZ7?usp=share_link
- Visium raw data and intermediate files: https://drive.google.com/drive/folders/1N2SJfrH7kDa3tjfcgwfZ4RbEJegAvXhT?usp=share_link (Starfysh deconvolution signature: `GVHD_spatial_signature_v8_major_curated_unique_epi_tcells_subset_Transition_EPI_REFINE.csv`)
- scRNA raw data and intermediate files: https://drive.google.com/drive/folders/1yV3ByTQL-CTCRNtnZf6tnRKFycLZseWp?usp=share_link (meta file: `scRNAseq_TCR_samples_meta_LS-15_f.xlsx`)
- MLR raw data and meta file: https://drive.google.com/drive/folders/1oyI_0J4Lk4-Sx1CYpHzdhNx-I0KqO4Bf?usp=sharing
- MLR PTcy analysis (5 additional patients) raw data and meta files: https://drive.google.com/drive/folders/1ftcSLg4Bx-BiXCFiQpUNSSJ-2spo8bAa?usp=drive_link
- DecompTCR input data: https://drive.google.com/drive/folders/1nWfx-3WkVEzTe4uYZl30jx9bezV8zHcX?usp=drive_link
- CDR3 amino acid analysis input: https://drive.google.com/drive/folders/1mRUPh0apwHe0LyPwnUqpOtrTvWDtoBcU?usp=drive_link

Figures

The table below maps each published figure to the notebook(s)/script(s) that generate it, following the layout used in [decipher_reproducibility](https://github.com/azizilab/decipher_reproducibility). Paths are relative to this repository's root. `MLR_Pipelines/README.md` and `DecipherTCR/README.md` contain additional per-panel detail for their pipelines.

> **Note on figure numbering:** a few notebooks contain leftover code comments referencing figure numbers that no longer match the published manuscript (e.g. `# Figure 4a`/`4B`/`4C`/`4F`/`4G` in `Sc&TCR_Pipelines/4.`, `6.`, `7.`; `# Figure 6A`/`Figure 7A` in the two `StarfyshHD` Starfysh notebooks; "supplementary figure 22" in `StarfyshHD/CellphoneDB_Analysis.ipynb`), which is one ahead of, or otherwise diverged from, the final published numbering — most likely from a mid-revision figure reorder. The mapping below follows the **published** figure numbers, verified against the specific genes, sample sizes, and statistical tests named in each caption rather than the in-code labels.

## Figure 1
- MLR_Pipelines/notebooks/01_patient_figures_and_diversity.Rmd (b)
- MLR_Pipelines/notebooks/02_clone_analysis_corrected_fig1C.Rmd (c)
- MLR_Pipelines/notebooks/04_cumulative_figures_corrected.Rmd (d)
- MLR_Pipelines/notebooks/03_grade_ptcy_analysis_corrected.Rmd (e–g)

## Figure 2
- DecompTCR/pipelines/basis_decomposition_generate.ipynb (a)
- DecompTCR/pipelines/basis_decomposition_final.ipynb (b)
- MLR_Pipelines/MLR_rcode_work_for_all_sample_combined_final_ptcy_github.Rmd (c–e)
- MLR_Pipelines/notebooks/05_figure3_annotation_variants.Rmd (f–h)

## Figure 3
- Sc&TCR_Pipelines/4. after annotation analysis.ipynb (a)
- Sc&TCR_Pipelines/6. T_cell_pipline_T_cell_subsets_signatures.ipynb (b, c, f)
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb (d, g, h)
- Sc&TCR_Pipelines/10.Donor_recipient_by_cell_type_freemuxlet.ipynb (e)

## Figure 4
- Sc&TCR_Pipelines/11.Migration_analysis.ipynb (a–f)
- DecipherTCR/2. decipher_ananlysis_GVHD_main_Tcells_CD8_conv_deg_4cluster_clean_with_figures_mobile_balanced.ipynb (g–i)

`DecipherTCR/1. decipher_ananlysis_GVHD_all_Tcells_all_genes_final.ipynb` is a prerequisite step (loads the raw AnnData and runs Decipher training) for notebook 2 above; it contains no `savefig` calls itself and does not directly produce any panel.

## Figure 5
- StarfyshHD/starfysh_ST_celltype_proportion_with_figures_clean.ipynb (a–c)
- StarfyshHD/starfysh_ST_distance_analysis.ipynb (a, d–f)

## Figure 6
- StarfyshHD/starfysh_ST_distance_analysis.ipynb (a, b)

Panel c is a non-code schematic (credit: SciStories).

## Supplementary Figure 1
- MLR_Pipelines/notebooks/01_patient_figures_and_diversity.Rmd

## Supplementary Figure 2
- MLR_Pipelines/notebooks/04_cumulative_figures_corrected.Rmd

## Supplementary Figure 3
- MLR_Pipelines/notebooks/02_clone_analysis_corrected_fig1C.Rmd

## Supplementary Figure 4
- MLR_Pipelines/notebooks/02_clone_analysis_corrected_fig1C.Rmd
- MLR_Pipelines/notebooks/03_grade_ptcy_analysis_corrected.Rmd

## Supplementary Figure 5
- DecompTCR/pipelines/basis_decomposition_generate.ipynb

## Supplementary Figure 6
- DecompTCR/pipelines/basis_decomposition_final.ipynb (a, b)
- DecompTCR/pipelines/tcrdist.ipynb (c)
- MLR_Pipelines/Basis_Hydrophobicity_CDR3Length_Figures.Rmd (d, e)

## Supplementary Figure 7
- DecompTCR/pipelines/basis_decomposition_final.ipynb

## Supplementary Figure 8
- MLR_Pipelines/MLR_rcode_work_for_all_sample_combined_final_ptcy_github.Rmd (a)
- MLR_Pipelines/notebooks/03_grade_ptcy_analysis_corrected.Rmd (b–f)

## Supplementary Figure 9
- MLR_Pipelines/notebooks/02_clone_analysis_corrected_fig1C.Rmd
- MLR_Pipelines/notebooks/03_grade_ptcy_analysis_corrected.Rmd

## Supplementary Figure 10
- Sc&TCR_Pipelines/4. after annotation analysis.ipynb

## Supplementary Figure 11
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb

## Supplementary Figure 12
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb

## Supplementary Figure 13
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb (a, c)
- Sc&TCR_Pipelines/9. Clonotype_T_cell_pipline_Migration.ipynb (b, e)
- Sc&TCR_Pipelines/11.Migration_analysis.ipynb (d)

## Supplementary Figure 14
- Sc&TCR_Pipelines/10.Donor_recipient_by_cell_type_freemuxlet.ipynb

## Supplementary Figure 15
- Sc&TCR_Pipelines/6. T_cell_pipline_T_cell_subsets_signatures.ipynb (b)
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb (a, d–g)

## Supplementary Figure 16
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb

## Supplementary Figure 17
- Sc&TCR_Pipelines/15a. Mapping_clones_between_MLR_and_tissue_TCRB_with_new_patients_all_Tcells.ipynb
- Sc&TCR_Pipelines/15b. Mapping_clones_between_MLR_and_tissue_TCRB_with_new_patients_map_to_donors.ipynb

## Supplementary Figure 18
- Sc&TCR_Pipelines/11.Migration_analysis.ipynb

## Supplementary Figure 19
- Sc&TCR_Pipelines/11.Migration_analysis.ipynb

## Supplementary Figure 20
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb

## Supplementary Figure 21
- Sc&TCR_Pipelines/13. CellTypist_analysis_for_T_cells_3_resi_cluster.ipynb

## Supplementary Figure 22
- Sc&TCR_Pipelines/12.Tcell_clonaltype_analysis_with_refined_cluster.ipynb

## Supplementary Figure 23
- Sc&TCR_Pipelines/TRB_clonality_validation.ipynb

## Supplementary Figure 24
- DecipherTCR/2. decipher_ananlysis_GVHD_main_Tcells_CD8_conv_deg_4cluster_clean_with_figures_mobile_balanced.ipynb

## Supplementary Figure 25
- DecipherTCR/2. decipher_ananlysis_GVHD_main_Tcells_CD8_conv_deg_4cluster_clean_with_figures_mobile_balanced.ipynb

## Supplementary Figure 26
- StarfyshHD/starfysh_ST_celltype_proportion_with_figures_clean.ipynb

## Supplementary Figure 27

`StarfyshHD/CellSAM_Segmentation.ipynb` is the notebook this figure should come from, but the file is currently empty (0 bytes) in this repository and will need to be restored.

## Supplementary Figure 28
- StarfyshHD/Starfysh_tutorial_integration_all_with_no_arch_with_figures_final_github.ipynb

## Supplementary Figure 29
- StarfyshHD/starfysh_ST_celltype_proportion_with_figures_clean.ipynb

## Supplementary Figure 30
- StarfyshHD/starfysh_ST_distance_analysis.ipynb (a, b)
- StarfyshHD/CellphoneDB_Analysis.ipynb (c)

## Supplementary Figure 31

Gating strategy for cell sorting (flow cytometry); not produced by a notebook in this repository.

## Supplementary Figure 32
- Sc&TCR_Pipelines/7. T_cell_pipline_proportion_analysis.ipynb

## Supplementary Figure 33
- Sc&TCR_Pipelines/11.Migration_analysis.ipynb

Acknowlegement

This work was made possible by the collaboration between the Azizi Lab and Reshef Lab, as well as our clinical partners at CUIMC and CCTI. We thank all participating patients and acknowledge the funding sources that supported this research.

Citation

Spatiotemporal Single-Cell Analysis Reveals T Cell Clonal Dynamics and Phenotypic Plasticity in Human Graft-versus-Host Disease
Lingting Shi, Ajna Uzuni, Ximi K. Wang, Michael Pressler, David W. Harle, Shami Chakrabarti, Rodney Macedo, Kirubel Belay, Christian A. Gordillo, Erik Raps, Jia Yi (Ady) Zhang, Achille Nazaret, Joy L. Fan, Yinuo Jin, Xumin Shen, Joshua S. Fuller, Tamjeed Azad, Jessie Huang, Pranik Chainani, Julian A. Abrams, Armando Del Portillo, Markus Y. Mapara, Mohamed Alhamar, Megan Sykes, José L. McFaline-Figueroa, Elham Azizi, Ran Reshef
bioRxiv 2025.05.24.655962; doi: https://doi.org/10.1101/2025.05.24.655962
