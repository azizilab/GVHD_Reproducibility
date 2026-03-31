# ============================================================================
# MLR Analysis - Central Configuration
# ============================================================================
# This file contains all paths, parameters, and package loading for the
# MLR analysis pipeline. Source this file at the beginning of each notebook.
#
# Usage: source("config/config.R")  # from notebooks/
#        source("../config/config.R")  # from notebooks/ subdirectory
# ============================================================================

cat("Loading MLR Analysis configuration...\n")

# ============================================================================
# CONDA R LIBRARY PATH
# ============================================================================
# When running via 'conda run -n MLR', the conda env's R library may not be
# in .libPaths() by default. Add it explicitly so conda-installed packages
# (e.g. cdr3tools) are found.
conda_prefix <- Sys.getenv("CONDA_PREFIX")
if (nchar(conda_prefix) > 0) {
  conda_r_lib <- file.path(conda_prefix, "lib", "R", "library")
  if (dir.exists(conda_r_lib) && !(conda_r_lib %in% .libPaths())) {
    .libPaths(c(conda_r_lib, .libPaths()))
  }
}

# ============================================================================
# PATHS
# ============================================================================

# Notebooks are rendered with wd set to their own directory (MLR_analysis_figure_1_2/notebooks/).
# All paths are computed relative to that working directory.
NOTEBOOKS_DIR    <- normalizePath(getwd())                          # .../MLR_analysis_figure_1_2/notebooks
FIGURE_1_2_DIR   <- normalizePath(file.path(NOTEBOOKS_DIR, ".."))  # .../MLR_analysis_figure_1_2
BASE_PATH        <- normalizePath(file.path(NOTEBOOKS_DIR, "../.."))# .../MLR_analysis

DATA_PATH          <- file.path(BASE_PATH,        "Annotated_v4")
EXCEL_PATH         <- file.path(FIGURE_1_2_DIR,   "TCR_Project.xlsx")
OUTPUT_PATH        <- file.path(FIGURE_1_2_DIR,   "Main_fig")
PROCESSED_DATA_PATH <- file.path(FIGURE_1_2_DIR,  "data/processed")
GOLD_STANDARD_PATH <- file.path(FIGURE_1_2_DIR,   "Main_fig_gold_standard")
PLOTS_DIR          <- FIGURE_1_2_DIR

# ============================================================================
# PARAMETERS
# ============================================================================

# Clone identification parameters
UNIQUE_IDENTIFIERS <- c("cdr3_amino_acid", "v_resolved", "j_resolved")

# Figure parameters
FIGURE_DPI <- 600
TEXT_SIZE <- 20
N_BOOTSTRAPS <- 1000

# Statistical parameters
ALPHA <- 0.05  # Significance level

# ============================================================================
# PACKAGE LOADING
# ============================================================================

required_packages <- c(
  # Immunoseq analysis
  "cdr3tools",
  # "Biostrings",  # Commented out in original notebook

  # Data manipulation
  "tidyverse",
  "dplyr",
  "purrr",
  "tibble",

  # Visualization
  "ggplot2",
  "gridExtra",
  "pheatmap",
  "patchwork",
  "ggalluvial",
  "ggtext",
  "plotly",
  "cowplot",
  "ggbeeswarm",
  "ggpubr",

  # Statistical analysis
  "rstatix",
  "tidymodels",

  # Diversity metrics
  "factoextra",
  "lvplot",

  # Utilities
  "readxl",
  "eulerr",
  "KneeArrower",
  "docstring"
)

# Load packages with suppressed startup messages
cat("Loading required packages...\n")
loaded_packages <- c()
failed_packages <- c()

suppressPackageStartupMessages({
  for (pkg in required_packages) {
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      loaded_packages <- c(loaded_packages, pkg)
    } else {
      failed_packages <- c(failed_packages, pkg)
      warning(paste0("Package '", pkg, "' is not available. Some features may not work."))
    }
  }
})

# Check for critical packages (absolutely required)
critical_packages <- c("cdr3tools", "tidyverse", "ggplot2", "dplyr", "readxl")
missing_critical <- setdiff(critical_packages, loaded_packages)
if (length(missing_critical) > 0) {
  stop(paste0("ERROR: Critical packages missing: ", paste(missing_critical, collapse=", "), "\n",
              "Please install them in the MLR conda environment."))
}

# ============================================================================
# R OPTIONS
# ============================================================================

# Set default PDF output for plots
options(device = function() {
  pdf(file = file.path(PLOTS_DIR, "Rplots.pdf"))
})

# Increase memory limit for large datasets
# options(max.print = 1000000)

# ============================================================================
# VERIFY PATHS
# ============================================================================

# Check that critical paths exist
if (!dir.exists(DATA_PATH)) {
  stop(paste("ERROR: Data directory not found:", DATA_PATH))
}

if (!file.exists(EXCEL_PATH)) {
  stop(paste("ERROR: Excel metadata file not found:", EXCEL_PATH))
}

# Create output directories if they don't exist
dir.create(OUTPUT_PATH, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_PATH, "Figure1"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_PATH, "Figure1ex"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_PATH, "Figure3"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_PATH, "Figure3ex"), showWarnings = FALSE, recursive = TRUE)
dir.create(PROCESSED_DATA_PATH, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# CONFIRMATION
# ============================================================================

cat("\n============================================================\n")
cat("MLR Analysis Configuration Loaded Successfully\n")
cat("============================================================\n")
cat("Data path:           ", DATA_PATH, "\n")
cat("Excel metadata:      ", EXCEL_PATH, "\n")
cat("Output path:         ", OUTPUT_PATH, "\n")
cat("Processed data path: ", PROCESSED_DATA_PATH, "\n")
cat("Gold standard path:  ", GOLD_STANDARD_PATH, "\n")
cat("Packages loaded:     ", length(required_packages), "\n")
cat("============================================================\n\n")
