#!/usr/bin/env Rscript
# run_all.R — Run all MLR analysis notebooks in sequence
# Usage: conda run -n MLR Rscript run_all.R

cat("=== MLR Analysis - Full Pipeline ===\n\n")

notebooks <- c(
  "notebooks/00_data_loading.Rmd",
  "notebooks/01_patient_figures_and_diversity.Rmd",
  "notebooks/02_clone_analysis.Rmd",
  "notebooks/03_grade_ptcy_analysis.Rmd",
  "notebooks/04_cumulative_figures.Rmd"
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
