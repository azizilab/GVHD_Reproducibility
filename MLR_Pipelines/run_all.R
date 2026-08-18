#!/usr/bin/env Rscript
# run_all.R — Run the full, verified MLR analysis pipeline in sequence.
# Usage: conda run -n MLR Rscript run_all.R
#
# Order matters: each notebook reads .rds files written by an earlier one.
# 00 loads raw immunoseq data and writes data/processed_corrected/; 01 (no
# corrected counterpart by design) writes 4 files back into data/processed/
# that 02-04 also depend on; 02 writes objects 03 reads for fig_EF; 04 and 05
# are independent terminal consumers.

cat("=== MLR Analysis - Full Pipeline ===\n\n")

notebooks <- c(
  "notebooks/00_data_loading_corrected.Rmd",
  "notebooks/01_patient_figures_and_diversity.Rmd",
  "notebooks/02_clone_analysis_corrected_fig1C.Rmd",
  "notebooks/03_grade_ptcy_analysis_corrected.Rmd",
  "notebooks/04_cumulative_figures_corrected.Rmd",
  "notebooks/05_figure3_annotation_variants.Rmd"
)

for (nb in notebooks) {
  cat(sprintf("Running: %s\n", nb))
  start <- proc.time()
  tryCatch({
    rmarkdown::render(nb, output_dir = "notebooks")
    elapsed <- proc.time() - start
    cat(sprintf("  PASSED (%.1fs)\n\n", elapsed["elapsed"]))
  }, error = function(e) {
    cat(sprintf("  FAILED: %s\n\n", conditionMessage(e)))
    stop(sprintf("Pipeline halted at %s", nb))
  })
}

cat("=== All notebooks completed successfully ===\n")
