# MLR Analysis Functions

This directory contains extracted function definitions from the MLR analysis pipeline.

## File: mlr_functions.R

A comprehensive collection of all functions used in the MLR analysis for T-cell receptor repertoire analysis in GVHD patients.

### Usage

To use these functions in your R notebooks or scripts:

```r
source("/home/user/Documents/GVHD_project/MLR_analysis/MLR_analysis_figure_1_2/functions/mlr_functions.R")
```

### Function Categories

#### 1. Data Processing Functions (6 functions)
- `combine_immuno()` - Combine immunoseq files from the same timepoint
- `determine_cell_type()` - Determine cell type based on unstimulated and CFSElo data
- `process_data()` - Process patient immunoseq folder and determine alloreactive clones
- `expand_counts()` - Expand clone counts for statistical analysis
- `downsample()` - Downsample immunoseq data to a specified number of templates

#### 2. Counting and Metrics Functions (6 functions)
- `count_unique_tags_by_patient()` - Count unique MLRCFSElo tags by patient ID
- `sum_template_by_patient()` - Sum template values by patient ID
- `count_unique_clones()` - Count unique clones per patient, timepoint, and combinations
- `calculate_clone_metrics()` - Calculate comprehensive clone metrics including diversity measures
- `count_clones()` - Bin clones over time for analysis

#### 3. Analysis Functions (8 functions)
- `cum_freq()` - Calculate cumulative frequency of alloreactive clones
- `avg_freq()` - Calculate average frequency of alloreactive clones
- `clone_frac()` - Calculate fraction of alloreactive clones
- `n_clone()` - Count number of clones
- `get_amino_arrangements()` - Get amino acid arrangements and frequencies
- `get_median_length()` - Calculate median CDR3 amino acid length
- `get_timepoint()` - Get earliest post-stimulation timepoints for each patient
- `get_early_timepoint()` - Get earliest and specific early timepoints for analysis
- `div_clones()` - Divide clones based on timepoints and diversity metrics

#### 4. Visualization Functions (8 functions)
- `GeomSplitViolin` - ggproto object for split violin plots
- `create_quantile_segment_frame()` - Create quantile segment frames for violin plots
- `geom_split_violin()` - Geom layer for split violin plots
- `div_fig()` - Create diversity metric figures over time
- `clone_fig()` - Visualize clone frequencies over time
- `num_clone_fig()` - Visualize number of clones over time with statistics
- `num_clones()` - Analyze and visualize number of clones with boxplots
- `avg_clone_fig()` - Visualize average clone frequency over time
- `div_cut()` - Analyze diversity metrics and visualize with statistical tests

#### 5. Utility Functions (2 functions)
- `add_tag()` - Add tag column combining CDR3, V, and J gene information
- `convert_id()` - Convert sample IDs to extract timepoint and patient information

### Total Functions: 30

### Dependencies

These functions require the following R packages:
- dplyr
- tidyr
- purrr
- ggplot2
- ggpubr
- rstatix
- ggbeeswarm
- scales
- plyr
- grid
- Biostrings (for amino acid analysis)
- ggtext (for markdown text in plots)

### Notes

- All functions include comprehensive documentation with parameter descriptions and usage examples
- Functions preserve all original comments and implementation details
- The file is organized into logical sections for easy navigation
- Line numbers and cross-references are maintained for traceability to the original notebook

### Author

Lingting

### Last Updated

2026-02-26
