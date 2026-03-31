# ==============================================================================
# MLR Analysis Functions
# ==============================================================================
#
# This file contains all function definitions used in the MLR analysis pipeline
# for T-cell receptor repertoire analysis in GVHD patients.
#
# Functions are organized into the following sections:
#   1. Data Processing Functions
#   2. Counting and Metrics Functions
#   3. Analysis Functions (diversity, frequency, etc.)
#   4. Visualization Functions
#   5. Utility Functions
#
# Author: Lingting
# Last Updated: 2026-02-26
# ==============================================================================


# ==============================================================================
# 1. DATA PROCESSING FUNCTIONS
# ==============================================================================

combine_immuno <- function(immdata, sample_meta_data, comb_idx, unique_identifiers){
  #' Combine immunoseq files from teh same timepoint
  #'
  #' @description This function combines n immunoseq
  #' files from the same timepoint that are given from Adaptive Imunno Sequencing.
  #' The file names must be in format PatientID_timepoint_celltype
  #'
  #' @param immdata List of dataframes containing the data from immunoseq files.
  #' Dataframes are obtained through the function read_immunoseq
  #' @param sample_meta_data 3 by sample dataframe that contains PatientID in first row,
  #' timepoint in second row, and cell type in the third row
  #' @param comb_idx the indices of the samples you would like to combine
  #' @param unique_identifiers Columns in immdata that identify unique clones (eg. cdr3_amino_acid, v_resolved, j_resolved)
  #' @usage combine_immuno(immdata, sample_meta_data, comb_idx, unique_identifiers)
  #' @return updated: list(immdata, sample_meta_data)



  cd4 <- immdata[[comb_idx[1]]] #initialize dataframe to be combined into
  cd4$cell_type<- sample_meta_data[3,comb_idx[1]] #set the cell_type of that dataframe
  for(i in seq(2,length(comb_idx),1)){ #for every other datafram identify by the indices combine into the first dataframe
    cd <- immdata[[comb_idx[i]]]
    cd$cell_type <- sample_meta_data[3, comb_idx[i]]
    cd4 <- dplyr::full_join(cd4,cd, by = c(unique_identifiers, "PatientID", "timepoint"))%>%
      replace_na(list(frequency.x = 0, frequency.y = 0,templates.x = 0, templates.y = 0, cell_type.x = "", cell_type.y = ""))
    cd4$templates <- cd4$templates.x+ cd4$templates.y #Add the templates of the same clones together
    cd4$frequency <- cd4$templates/sum(cd4$templates) #Recalculate the frequency
    cd4$cell_type <- gsub(" ", "",paste(gsub(" ", "",cd4$cell_type.x), gsub(" ", "", cd4$cell_type.y)))
    cd4 <- cd4 %>% select(all_of(c(unique_identifiers, "frequency", "templates", "PatientID", "timepoint", "cell_type")))
    immdata[[comb_idx[1]]] <- cd4
  }
  sample_meta_data[3,comb_idx[1]] <- "both" #change the first dataframes description to be both
  comb_idx <- comb_idx[-1] #remove the first indices from the indices
  immdata <- immdata[-comb_idx] #remove the dataframes that were combined from immdata
  sample_meta_data <- sample_meta_data[,-comb_idx] #remove the samples that were combined from sample_meta_data

  return(list(immdata, sample_meta_data))

}


determine_cell_type <- function(cell_type.unstim, cell_type.CFSElo){
  if(cell_type.unstim == "CD4CD8" & cell_type.CFSElo != "CD4CD8"){
    return (cell_type.CFSElo)
  } else if(cell_type.CFSElo == "CD4CD8" & cell_type.unstim != "CD4CD8"){
    return (cell_type.unstim)
  } else if(cell_type.unstim != "CD4CD8" & cell_type.unstim != "CD4CD8"){
    if(cell_type.CFSElo == cell_type.unstim)
    {
      return(cell_type.CFSElo)
    }
    else
    {
      return("CD4CD8")
    }
  }

  return("CD4CD8")
}


process_data <- function(filename, path, unique_identifiers = c("cdr3_amino_acid", "v_resolved", "j_resolved")){
  #' Process patient immunoseq folder
  #'
  #' @description This function takes a patients immunoseq folder and iterates
  #' through each file extracting the needed information and combining files that
  #' are from the same timepoint. Once files are combined alloreactive clones
  #' are determined and returned.
  #'
  #' @param filename name of patient folder
  #' @param path to folder that contains all patients
  #' @param unique_identifiers Columns in immdata that identify unique clones (eg. cdr3_amino_acid, v_resolved, j_resolved)
  #' @return list(immdata2, sample_meta_data, allo, nonallo)
  #'

  # load data
  print(filename)

  # Construct full path to patient directory
  patient_dir <- file.path(path, filename)
  cat("Reading from:", patient_dir, "\n")

  # Read immunoseq data from patient directory
  immdata <- cdr3tools::read_immunoseq(patient_dir)
  immdata2 <- lapply(immdata, function(x) {
  x %>%
    dplyr::group_by(across(all_of(unique_identifiers))) %>% #Combine any duplicates and drop any unnecessary rows
    dplyr::summarise(templates = sum(templates), .groups = "drop") %>%
    mutate(frequency = templates / sum(templates)) %>%
    dplyr::arrange(dplyr::desc(frequency))
    })


  immdata1 <- map2( #Map PatientID to clones
    immdata2,
    data.frame(strsplit(names(immdata), split = "_"))[1,],
    ~ mutate(.x, PatientID = .y)
  )
  immdata2 <- map2( #map time to each sample
    immdata1,
    data.frame(strsplit(names(immdata), split = "_"))[2,],
    ~ mutate(.x, timepoint = .y)
  )

  # get metadata
  sample_names = rownames(summary(immdata))
  sample_meta_data <-data.frame(strsplit(sample_names, split = "_"))
  sample_meta_data


  Treg_idx <- grep('Treg', sample_meta_data[3,]) #Combine Treg
  if(length(Treg_idx) > 1){
    result <- combine_immuno(immdata2, sample_meta_data, Treg_idx, unique_identifiers)
    immdata2 <- result[[1]]
    sample_meta_data <- result[[2]]
  }

  MAX_ITER <-50 #may need to change if you are combining over 50 samples
  iter <- 0
  j <- 1
  while(j <= length(sample_meta_data[2,]) & iter < MAX_ITER){ #Combine and CD4 and CD8 that have the same timepoints

    time_idx <- which(sample_meta_data[2,] == sample_meta_data[2,j])
    if (length(time_idx) > 1){ #If there is one CD4 or CD8 Change to both
      result <- combine_immuno(immdata2, sample_meta_data, time_idx, unique_identifiers)
      immdata2 <- result[[1]]
      sample_meta_data <- result[[2]]
      #Ensure Cell type labeling is good
      # immdata2[[j]] <- immdata2[[j]] %>%
      #   mutate(cell_type = ifelse(grepl("CD3", cell_type), gsub("CD3 & CD8")))

    }else{
      if (sample_meta_data[3,j] != "both") {
        cd48 <- immdata2[[j]]
        cd48 <- cd48 %>%
          subset(select = c(unique_identifiers, "frequency", "templates", "PatientID", "timepoint"))
        cd48$cell_type <- sample_meta_data[3,j]
        immdata2[[j]] <- cd48
        sample_meta_data[3,j] = "both"
      }else{
        #print("Already Both")
        immdata2[[j]]$cell_type = "CD4CD8"
      }
    }
    iter <- iter + 1
    j <- j + 1
  }

  unstim_idx <- which(sample_meta_data[2,] == 'unstimulated')
  MLRCF_idx <- which(sample_meta_data[2,] == 'MLRCFSElo')

  allo_inner <- dplyr::inner_join(immdata2[[MLRCF_idx]],immdata2[[unstim_idx]],
                           by = c(unique_identifiers, "PatientID"),
                           suffix = c(".CFSELo", ".unstim")
  ) %>%
    replace_na(list(frequency.unstim = 0, frequency.CFSELo = 0)) %>%
    filter(frequency.CFSELo >= 2 * frequency.unstim)%>%
    select(unique_identifiers, "frequency.CFSELo", "frequency.unstim", "templates.CFSELo", "templates.unstim", "PatientID", "cell_type.CFSELo")
  allo_inner['tag'] <- 'inner'
  colnames(allo_inner)[colnames(allo_inner) == "cell_type.CFSELo"] <- "cell_type"

  allo_outer <- dplyr::anti_join(immdata2[[MLRCF_idx]],immdata2[[unstim_idx]],
                          by = c(unique_identifiers),
  ) %>%
    replace_na(list(frequency = 0)) %>%
    filter(frequency >= 1e-5)%>%
    select(unique_identifiers, "frequency", "templates", "PatientID", "cell_type")
  names(allo_outer)[names(allo_outer) == 'frequency'] <- 'frequency.CFSELo'
  names(allo_outer)[names(allo_outer) == 'templates'] <- 'templates.CFSELo'
  allo_outer['frequency.unstim'] <- 0
  allo_outer['templates.unstim'] <- 0
  allo_outer['tag'] <- 'outer'
  allo <- bind_rows(allo_inner, allo_outer)

  nonallo_unstim <- dplyr::anti_join(immdata2[[unstim_idx]], allo, by = c(unique_identifiers))%>%
    replace_na(list(frequency = 0))%>%
    select(unique_identifiers, "frequency", "templates", "PatientID", "cell_type")

  nonallo_mlr <- dplyr::anti_join(immdata2[[MLRCF_idx]], allo, by = c(unique_identifiers))%>%
    replace_na(list(frequency = 0))%>%
    select(unique_identifiers, "frequency", "templates", "PatientID", "cell_type")

  nonallo <- dplyr::full_join(nonallo_unstim, nonallo_mlr,
                              by = c(unique_identifiers, "PatientID"),
                              suffix = c(".unstim", ".CFSElo"))%>%
    replace_na(list(frequency.unstim = 0, frequency.CFSElo = 0,
                    templates.unstim = 0, templates.CFSElo = 0))%>%
    mutate(cell_type.unstim = gsub("CD3", "", cell_type.unstim), cell_type.CFSElo = gsub("CD3", "", cell_type.CFSElo))%>%
    mutate(cell_type.unstim = ifelse(is.na(cell_type.unstim), cell_type.CFSElo, cell_type.unstim), cell_type.CFSElo = ifelse(is.na(cell_type.CFSElo), cell_type.unstim, cell_type.CFSElo))%>%
    mutate(cell_type.CFSElo = gsub("CD3", "", cell_type.CFSElo ), cell_type.unstim = gsub("CD3","", cell_type.unstim))%>%
    mutate(cell_type.CFSElo = gsub("both", "CD4CD8", cell_type.CFSElo ), cell_type.unstim = gsub("both","CD4CD8", cell_type.unstim))%>%
    mutate(cell_type.CFSElo = gsub("tsv", "CD4CD8", cell_type.CFSElo ), cell_type.unstim = gsub("tsv","CD4CD8", cell_type.unstim))%>%
    mutate(cell_type.CFSElo = ifelse(cell_type.CFSElo == "", "CD4CD8", cell_type.CFSElo), cell_type.unstim = ifelse(cell_type.unstim == "", "CD4CD8", cell_type.unstim))%>%
    rowwise()%>%
    mutate(cell_type = determine_cell_type(cell_type.CFSElo, cell_type.unstim))
  # %>%
  #   mutate(cell_type = paste(ifelse((cell_type.unstim), "", cell_type.unstim), ifelse(is.na(cell_type.CFSElo), "", cell_type.CFSElo)))%>%
  #   mutate(cell_type = paste(ifelse(grepl("8",cell_type), "CD8", ""),ifelse(grepl("4",cell_type), "CD4", ""), ifelse(grepl("3",cell_type), "CD3", "") ))#%>%
  #   #select(all_of(c(unique_identifiers, "cell_type")))

  return(list(immdata2, sample_meta_data, allo, nonallo))
}


expand_counts <- function(data) {

  # Create an empty list to store the expanded data frames
  expanded_data_list <- list()

  # Loop through each row of the data frame
  for (i in seq_len(nrow(data))) {
    # Extract the id and count for the current row
    cdr3 <- data$cdr3_amino_acid[i]
    v <- data$v_resolved[i]
    j <- data$j_resolved[i]
    count <- data$templates[i]

    # Create a data frame with 'count' rows and a single 1 in the 'count' column
    expanded_data <- data.frame(cdr3_amino_acid = rep(cdr3, count),v_resolved = rep(v, count),j_resolved = rep(j, count), templates = rep(1, count))

    # Add the expanded data frame to the list
    expanded_data_list[[i]] <- expanded_data
  }

  # Combine all the expanded data frames into a single data frame
  expanded_data <- do.call(rbind, expanded_data_list)

  # Return the expanded data frame
  return(expanded_data)
}


downsample <- function(immdata, num){
  immdata_exp <- expand_counts(immdata)
  immdata_exp_samp <- sample_n(immdata_exp, num)%>%
    group_by(cdr3_amino_acid, v_resolved, j_resolved)%>%
    summarise(templates = sum(templates), .groups="drop")%>%
    mutate(frequency = templates/sum(templates))

  return(immdata_exp_samp)

}


# ==============================================================================
# 2. COUNTING AND METRICS FUNCTIONS
# ==============================================================================

count_unique_tags_by_patient <- function(data_allo) {
  # Create a data frame to store results
  results <- data.frame(
    patient_id = character(),
    num_unique_tags = integer(),
    stringsAsFactors = FALSE
  )

  # Loop through each element in data_allo
  for (i in seq_along(data_allo)) {
    # Get all names at this level that end with "_MLRCFSElo_both"
    mlrcfselo_names <- grep("MLRCFSElo", names(data_allo[[i]]), value = TRUE)

    for (mlrcfselo_name in mlrcfselo_names) {
      # Extract patient ID from the name (assuming format like "R051_MLRCFSElo_both")
      patient_id <- sub("^([^_]+)_.*$", "\\1", mlrcfselo_name)

      # Get the data
      mlrcfselo_data <- data_allo[[i]][[mlrcfselo_name]]

      # Count unique tags based on data type
      num_unique <- 0
      if (is.data.frame(mlrcfselo_data) || is.matrix(mlrcfselo_data)) {
        # For tibbles/data frames, look for tag columns
        if ("tag" %in% colnames(mlrcfselo_data)) {
          num_unique <- length(unique(mlrcfselo_data$tag))
        } else {
          # If no obvious tag column, count rows as a fallback
          num_unique <- nrow(mlrcfselo_data)
        }
      } else if (is.vector(mlrcfselo_data)) {
        # For vectors, count unique elements
        num_unique <- length(unique(mlrcfselo_data))
      }

      # Add to results
      results <- rbind(results, data.frame(
        patient_id = patient_id,
        num_unique_tags = num_unique,
        stringsAsFactors = FALSE
      ))
    }
  }

  # Aggregate by patient_id (in case the same patient appears multiple times)
  if (nrow(results) > 0) {
    agg_results <- aggregate(num_unique_tags ~ patient_id, data = results, FUN = sum)
    return(agg_results)
  } else {
    return(results)
  }
}


sum_template_by_patient <- function(data_all) {
  # Create a data frame to store results
  results <- data.frame(
    patient_id = character(),
    sum_template = numeric(),
    stringsAsFactors = FALSE
  )

  # Loop through each element in data_allo
  for (i in seq_along(data_allo)) {
    # Get all names at this level that end with "_MLRCFSElo_both"
    mlrcfselo_names <- grep("MLRCFSElo", names(data_allo[[i]]), value = TRUE)

    for (mlrcfselo_name in mlrcfselo_names) {
      # Extract patient ID from the name (assuming format like "R051_MLRCFSElo_both")
      patient_id <- sub("^([^_]+)_.*$", "\\1", mlrcfselo_name)

      # Get the data
      mlrcfselo_data <- data_allo[[i]][[mlrcfselo_name]]

      # Sum template values based on data type
      template_sum <- 0
      if (is.data.frame(mlrcfselo_data) || is.matrix(mlrcfselo_data)) {
        # For tibbles/data frames, look for template column
        if ("templates" %in% colnames(mlrcfselo_data)) {
          # Sum the template column, handling NA values
          template_sum <- sum(mlrcfselo_data$templates, na.rm = TRUE)
        }
      }

      # Add to results
      results <- rbind(results, data.frame(
        patient_id = patient_id,
        sum_template = template_sum,
        stringsAsFactors = FALSE
      ))
    }
  }

  # Aggregate by patient_id (in case the same patient appears multiple times)
  if (nrow(results) > 0) {
    agg_results <- aggregate(sum_template ~ patient_id, data = results, FUN = sum)
    return(agg_results)
  } else {
    return(results)
  }
}


count_unique_clones <- function(data_all_squished) {
  # Ensure data is properly formatted
  if (!all(c("cdr3_amino_acid", "v_resolved", "j_resolved", "PatientID", "timepoint") %in% colnames(data_all_squished))) {
    stop("Required columns not found in dataset")
  }

  # 1. Count unique clones per patient
  clones_per_patient <- data_all_squished %>%
    group_by(PatientID) %>%
    summarise(
      unique_clones = n_distinct(paste(cdr3_amino_acid, v_resolved, j_resolved)),
      .groups = "drop"
    ) %>%
    arrange(PatientID)

  # 2. Count unique clones per timepoint
  clones_per_timepoint <- data_all_squished %>%
    group_by(timepoint) %>%
    summarise(
      unique_clones = n_distinct(paste(cdr3_amino_acid, v_resolved, j_resolved)),
      .groups = "drop"
    ) %>%
    arrange(timepoint)

  # 3. Count unique clones per patient and timepoint
  clones_per_patient_timepoint <- data_all_squished %>%
    group_by(PatientID, timepoint) %>%
    summarise(
      unique_clones = n_distinct(paste(cdr3_amino_acid, v_resolved, j_resolved)),
      .groups = "drop"
    ) %>%
    arrange(PatientID, timepoint)

  # Return all results as a list
  return(list(
    by_patient = clones_per_patient,
    by_timepoint = clones_per_timepoint,
    by_patient_timepoint = clones_per_patient_timepoint
  ))
}


calculate_clone_metrics <- function(data_all_squished) {
  # Basic counts
  clone_metrics <- data_all_squished %>%
    group_by(PatientID, timepoint) %>%
    summarise(
      total_templates = sum(templates, na.rm = TRUE),
      unique_clones = n_distinct(paste(cdr3_amino_acid, v_resolved, j_resolved)),
      unique_cdr3 = n_distinct(cdr3_amino_acid),
      unique_v_genes = n_distinct(v_resolved),
      unique_j_genes = n_distinct(j_resolved),
      .groups = "drop"
    )

  # Calculate summary per patient (across all timepoints)
  patient_summary <- data_all_squished %>%
    group_by(PatientID) %>%
    summarise(
      total_templates = sum(templates, na.rm = TRUE),
      unique_clones = n_distinct(paste(cdr3_amino_acid, v_resolved, j_resolved)),
      unique_cdr3 = n_distinct(cdr3_amino_acid),
      unique_v_genes = n_distinct(v_resolved),
      unique_j_genes = n_distinct(j_resolved),
      timepoints = n_distinct(timepoint),
      .groups = "drop"
    )

  return(list(
    detailed = clone_metrics,
    patient_summary = patient_summary
  ))
}


count_clones <- function(data, type, metric = "cum_freq", num_cuts = 5, group_by = c("PatientID","timepoint", "PTCy")){
  #'
  #' @description Bins the clones over time.
  #'
  #' @param data List of dataframes containing the data from immunoseq files.
  #' Dataframes are obtained through the function read_immunoseq
  #' @param type Grouping you would like to use
  #' @param metric diverseity metric you would like to use
  #' @param num_cuts, the number of bins you would like
  #'
  #'
  #' @usage count_clones(data, type, metric = "cum_freq", num_cuts = 5)
  #' @return updated: list(data_pre, data_post)
  #'
  #'
  data_proc <- data%>%
    group_by(across(all_of(c(group_by, type))))%>%
    dplyr::summarise(num_clones = sum(frequency), .groups = "keep")

  data_pre <- data_proc[grepl("MLR|unstim", data_proc$timepoint),] #Seperate timepoints before transplant

  data_proc <- data_proc %>%
    mutate(timepoint = strtoi(timepoint))
  data_proc <- data_proc[!is.na(data_proc$timepoint),]


  data_post <- data_proc[data_proc$timepoint <= 365 & data_proc$timepoint > 3,] %>%
    subset(!is.na(timepoint))%>%
    ungroup()

  qs <- matrix(nrow = length(unique(data_post[[type]])), ncol = num_cuts+1)
  for(i in seq_along(unique(data_post[[type]]))){
    t <- unique(data_post[[type]])[[i]]
    qs[i,] <- quantile(data_post[data_post[[type]] == t,]$timepoint, probs = 0:num_cuts/num_cuts)
  }

  qs_breaks <- apply(qs,2, max)

  breaks <- unique(quantile(data_post$timepoint, probs = 0:num_cuts/num_cuts))

  data_post <- data_post%>%
    mutate(bins = cut(timepoint,breaks = breaks, include.lowest = TRUE))

  return(list(data_pre, data_post))
}


# ==============================================================================
# 3. ANALYSIS FUNCTIONS (diversity, frequency, etc.)
# ==============================================================================

cum_freq <- function(allo_data, data){
  freqs <-as.data.frame(do.call(rbind, Map(function(x, y) {
    sum(x$frequency) / sum(y$frequency)
  }, x = allo_data, y = data)))
  return(freqs)
}


avg_freq <- function(allo_data, data){
  freqs <-as.data.frame(do.call(rbind, Map(function(x, y) {
    mean(x$frequency) / sum(y$frequency)
  }, x = allo_data, y = data)))
  return(freqs)
}


clone_frac <- function(allo_data, data){
  as.data.frame(do.call(rbind, Map(function(x, y) {
    nrow(x) / nrow(y)
  }, x = allo_data, y = data)))
}


n_clone <- function(allo_data, data){
  n_clones <- as.data.frame(do.call(rbind, Map(function(x, y) {
    nrow(x)
  }, x = allo_data, y = data)))
}


get_amino_arrangements <- function(immdata){#}, sample_meta_data){

  arrangements <- lapply(immdata, function(x){
    x %>%
      select("cdr3_amino_acid", "frequency") %>%
      mutate(cdr3_aminolength = nchar(cdr3_amino_acid))
  })

  arrangements <- bind_rows(arrangements, .id = "id")
  arrangements$timepoint <- str_split_fixed(arrangements$id,"_",3)[,2]
  arrangements$PatientID <- str_split_fixed(arrangements$id,"_",3)[,1]


  aa <- AAStringSet(arrangements$cdr3_amino_acid)
  freqs <- data.frame(alphabetFrequency(aa, as.prob = FALSE))

  arrangements <- bind_cols(arrangements, freqs)

  return(arrangements)
}


get_median_length <- function(arrangements, name){
  med_lengths <- arrangements %>%
    group_by(timepoint, PatientID)%>%
    dplyr::summarise(med_cdr3aminolength = mean(cdr3_aminolength), .groups = "drop")
  med_lengths[['Rep_div']] <- name

  return(med_lengths)
}


get_timepoint <- function(df) {
  #' Get earliest post-stimulation timepoints for each patient
  #'
  #' @description This function selects all timepoints less than or equal to 3 and the earliest timepoint greater than 3 for each patient.
  #'
  #' @param df A dataframe containing `timepoint` and `PatientID` columns.
  #' @return A dataframe with the earliest timepoints for each patient.

  df <- df %>%
    mutate(timepoint = strtoi(timepoint)) %>%
    filter(!is.na(timepoint))

  # Select all timepoints less than or equal to 3
  df_res <- df %>%
    filter(timepoint <= 3)

  # For each patient, select the earliest timepoint greater than 3
  df_patients <- df %>%
    group_by(PatientID) %>%
    filter(timepoint > 3) %>%
    filter(timepoint == min(timepoint)) %>%
    ungroup()

  # Combine the selected timepoints
  df_res <- bind_rows(df_res, df_patients)

  return(df_res)
}


get_early_timepoint <- function(df) {
  #' Get earliest and specific early timepoints for each patient
  #'
  #' @description This function selects the earliest timepoint greater than 3 and also specific early timepoints between 2 and 3 for each patient.
  #'
  #' @param df A dataframe containing `timepoint` and `PatientID` columns.
  #' @return A dataframe with the earliest timepoints and specific early timepoints for each patient.

  df_res <- data.frame()

  for (patient in unique(df$PatientID)) {
    # Select earliest timepoint greater than 3 for each patient
    df_pat <- df %>%
      filter(PatientID == patient, timepoint > 3) %>%
      filter(timepoint == min(timepoint))

    # Select timepoints between 2 and 3 for each patient
    df_pat_3 <- df %>%
      filter(PatientID == patient, timepoint >= 2, timepoint <= 3)

    # Combine the selected timepoints
    df_res <- bind_rows(df_res, df_pat, df_pat_3)
  }

  return(df_res)
}


div_clones <- function(data, type, div_metric = "shannons.entropy", num_cuts = 5){
  #' Divide clones based on timepoints and diversity metric
  #'
  #' @description This function processes a dataset of clones to divide it into pre- and post-stimulation data based on specific timepoints. The function filters and bins data to analyze clonal diversity using a specified diversity metric.
  #'
  #' @param data A dataframe containing clone data, including a `SampleID` column which indicates timepoints or conditions.
  #' @param type A character string specifying the type of analysis or dataset being processed.
  #' @param div_metric A character string specifying the diversity metric to use (default is "shannons.entropy").
  #' @param num_cuts An integer indicating the number of bins to divide the timepoints into (default is 5).
  #' @usage div_clones(data, type, div_metric, num_cuts)
  #' @return A list containing two dataframes: `data_pre` for pre-stimulation and `data_post` for post-stimulation data.

  data_pre <- data[grepl("MLR|unstim", data$SampleID),]%>%
    dplyr::mutate(timepoint = SampleID) # Filter for pre-stimulation samples and mutate timepoint to be the same as SampleID

  data_proc <- data %>%
    dplyr::mutate(timepoint = strtoi(SampleID)) # Convert SampleID to an integer timepoint for processing
  data_proc <- data_proc[!is.na(data_proc$timepoint),] # Remove any rows where timepoint conversion was not possible (NA values)

  data_post <- data_proc[data_proc$timepoint <= 365 & data_proc$timepoint > 3,] %>%
    ungroup() %>%
    dplyr::mutate(bins = cut(timepoint, breaks = unique(quantile(timepoint, probs = 0:num_cuts/num_cuts)), include.lowest = TRUE)) # Filter for post-stimulation samples within a year and create bins based on timepoints

  return(list(data_pre, data_post)) # Return a list containing pre- and post-stimulation dataframes
}


# ==============================================================================
# 4. VISUALIZATION FUNCTIONS
# ==============================================================================

# Custom geom for split violin plots
GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin,
  draw_group = function(self, data, ..., draw_quantiles = NULL) {
    # Original function by Jan Gleixner (@jan-glx)
    # Adjustments by Wouter van der Bijl (@Axeman)
    data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
    grp <- data[1, "group"]
    newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
    newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
    newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
    if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
      stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <= 1))
      quantiles <- create_quantile_segment_frame(data, draw_quantiles, split = TRUE, grp = grp)
      aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
      aesthetics$alpha <- rep(1, nrow(quantiles))
      both <- cbind(quantiles, aesthetics)
      quantile_grob <- GeomPath$draw_panel(both, ...)
      ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
    }
    else {
      ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
    }
  }
)


create_quantile_segment_frame <- function(data, draw_quantiles, split = FALSE, grp = NULL) {
  dens <- cumsum(data$density) / sum(data$density)
  ecdf <- stats::approxfun(dens, data$y)
  ys <- ecdf(draw_quantiles)
  violin.xminvs <- (stats::approxfun(data$y, data$xminv))(ys)
  violin.xmaxvs <- (stats::approxfun(data$y, data$xmaxv))(ys)
  violin.xs <- (stats::approxfun(data$y, data$x))(ys)
  if (grp %% 2 == 0) {
    data.frame(
      x = ggplot2:::interleave(violin.xs, violin.xmaxvs),
      y = rep(ys, each = 2), group = rep(ys, each = 2)
    )
  } else {
    data.frame(
      x = ggplot2:::interleave(violin.xminvs, violin.xs),
      y = rep(ys, each = 2), group = rep(ys, each = 2)
    )
  }
}


geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ...,
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE,
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, position = position,
        show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}


div_fig <- function(patient_data, div, Patient){
  #selected <- subset(patient_data, patient_data$Rep_div %in% types)
  fig <- ggplot(patient_data, aes(SampleID,.data[[div]], group = Rep_div, color=Rep_div,shape = Rep_div)) +
    geom_point(size = 1) +
    geom_line() +
    scale_y_continuous(trans='log10', labels = scales::number_format(accuracy = 0.01))+
    labs(y = div, x = "Days Post Transplantation", title = Patient) +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 45, vjust = 0.5, size = 8))
  return(fig)
}


clone_fig <- function(immdata, type, title, alpha = 0.3) {
  #' Visualize clone frequencies over time
  #'
  #' @description This function visualizes the frequency of clones over time for different groups using scatter and line plots.
  #'
  #' @param immdata A dataframe containing immunosequencing data with `timepoint`, `frequency`, `tag`, and specified `type`.
  #' @param type A character string specifying the grouping variable for the analysis.
  #' @param title A character string for the plot title.
  #' @param alpha A numeric value indicating the transparency level of points and lines in the plot (default is 0.3).
  #' @usage clone_fig(immdata, type, title, alpha)
  #' @return A ggplot object showing the clone frequency over time.

  # Convert timepoints to integer and remove NA values
  immdata <- immdata %>%
    mutate(timepoint = strtoi(timepoint)) %>%
    filter(!is.na(timepoint))

  # Calculate average and standard deviation of frequency per group and timepoint
  immdata_stat <- immdata %>%
    group_by(across(all_of(c(type, "timepoint")))) %>%
    summarise(
      avg_freq = mean(frequency, na.rm = TRUE),
      std_freq = sd(frequency, na.rm = TRUE),
      .groups = 'drop'
    )

  # Generate the plot
  fig <- ggplot(immdata, aes(x = timepoint, y = frequency, color = .data[[type]], group = tag)) +
    geom_point(size = 1, alpha = alpha) +
    geom_line(alpha = alpha) +
    labs(title = title, x = "Timepoint", y = "Frequency") +
    scale_y_continuous(trans = "log10", labels = scales::number_format(accuracy = 0.00001)) +
    scale_colour_discrete(na.translate = FALSE) +
    scale_shape_discrete(na.translate = FALSE) +
    guides(color = guide_legend(override.aes = list(alpha = 1))) +
    theme_pubr() +
    theme(
      axis.text.x = element_text(angle = 60, vjust = 0.5),
      legend.text = element_markdown(size = 20)
    )

  return(fig)
}


num_clone_fig <- function(immdata, type, title = "", num_cuts = 5, groupings = NA, colors = c("#E6BE60", "#A483AF"), method = "hochberg", legend_title = "", ylab = ""){
  #' Visualize the number of clones over time
  #'
  #' @description This function calculates and visualizes the number of clones over time, grouping by specified categories and applying statistical tests to compare groups.
  #'
  #' @param immdata A dataframe containing immunosequencing data with `timepoint`, `PatientID`, and specified `type`.
  #' @param type A character string specifying the grouping variable for the analysis.
  #' @param title A character string for the plot title.
  #' @param num_cuts An integer indicating the number of bins to divide the timepoints into (default is 5).
  #' @param groupings Optional grouping variable for pairwise comparisons (default is NA).
  #' @param colors A character vector specifying colors for the plot (default is c("#E6BE60", "#A483AF")).
  #' @param method A character string specifying the p-value adjustment method (default is "hochberg").
  #' @param legend_title A character string for the legend title.
  #' @param ylab A character string for the y-axis label in the plot.
  #' @usage num_clone_fig(immdata, type, title, num_cuts, groupings, colors, method, legend_title, ylab)
  #' @return A list containing the plots, statistical test results, and processed data.

  immdata$timepoint <- as.character(immdata$timepoint) #Ensure timepoints are characters

  immdata <- immdata%>%
    mutate(timepoint = strtoi(timepoint)) #Convert back to integers
  immdata <- immdata[!is.na(immdata$timepoint),] #remove any NA timepoints
  immdata <- immdata[immdata$timepoint <= 365,]
  immdata_3 <- immdata[immdata$timepoint <= 3,]%>%
    mutate(timepoint = 3, bins = "3")%>%
    dplyr::filter(templates != 0)
  immdata <- immdata[immdata$timepoint > 3,]
  immdata_all_timepoint <- immdata %>% #This is to keep track of timepoints that do not have any expanded clones
    select(all_of(c("timepoint","PatientID", type)))%>%
    distinct()

  immdata <- immdata[immdata$templates !=0, ]

  breaks <- unique(quantile(immdata_all_timepoint$timepoint, probs = 0:num_cuts/num_cuts, na.rm = T)) #Breaks based of all timepoints recorded
  print(breaks)

  immdata <- immdata%>%
    mutate(bins = cut(immdata$timepoint, breaks = breaks, include.lowest = TRUE))


  immdata_all_timepoint <- immdata_all_timepoint %>%
    mutate(bins = cut(immdata_all_timepoint$timepoint, breaks = breaks, include.lowest = TRUE))

  if(length(immdata_3$timepoint) != 0){
    immdata <- bind_rows(immdata_3, immdata)%>%
      mutate(bins = factor(bins, level = c("3", levels(immdata_all_timepoint$bins))))
  }else{
    immdata <- immdata%>%
      mutate(bins = factor(bins, level = levels(immdata_all_timepoint$bins)))
  }

  immdata_fig <- immdata%>%
    group_by(across(all_of(c("bins","timepoint","PatientID", type))))%>%
    tally()%>%
    ungroup()%>%
    dplyr::full_join(immdata_all_timepoint, by =c("PatientID", "bins","timepoint", type), multiple = "first")%>%
    replace_na(list(n = 0))%>%
    group_by(across(all_of(c("bins", type))))

  immdata_sum <- immdata_fig%>% ungroup()%>%
    group_by(across(all_of(c("bins", type))))%>%
    dplyr::summarise(m = mean(n), s = sqrt(var(n)))


  stat_test <- immdata_fig %>%
    group_by(bins) %>%
    mutate(n_in = n())%>%
    filter(n_in >= 3, sum(n != 0) > 0)%>% #dont do statistical test on groups that dont have enough samples
    pairwise_wilcox_test(as.formula(paste0(c("n ~ ",type))), p.adjust.method = "hochberg", exact = F, detailed = T)%>%
    add_xy_position(x = "bins")%>%
    ungroup()%>%
    mutate(
      y.position = log10(y.position+1),
      p.adj = p.adjust(p, method = "hochberg"),
      p.adj.signif = symnum(p.adj,cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1),
                            symbols = c("****", "***", "**", "*", "ns"))
      )


  fig <- ggplot(immdata_sum, aes(x = bins, y = m, fill = .data[[type]], group = .data[[type]], color = .data[[type]])) +
    geom_line()+
    geom_point()+
    geom_ribbon(aes(ymin = m - s, ymax = m + s, linetype = NA), alpha = 0.3)+
    geom_errorbar(aes(ymin = m - s, ymax = m + s), width = 0.1)+
    scale_x_discrete(na.translate = F)+
    theme_minimal()+
    theme(axis.text = element_text(size = 18),
          axis.text.x = element_text(angle = 75, vjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          axis.ticks = element_line(color = "black"),
          axis.ticks.length = unit(0.25, "cm"))+
    labs(title = title, y = ylab, x = "Days")+
    scale_fill_manual(values=colors)+
    scale_color_manual(values=colors)

  fig_vio <- ggboxplot(immdata_fig, x = "bins", y = "n", fill = type, outlier.shape = NA)+
    geom_beeswarm(data = immdata_fig[immdata_fig$n != 0,], dodge.width = 0.75, cex = 0.1, corral.width = 0.2, method = "compactswarm", aes(x= bins, y = n, group = .data[[type]]))+
    stat_pvalue_manual(stat_test, label = "p.adj.signif", tip.length = 0.01, step.increase = 0.00, size = 8, bracket.nudge.y = -0.15)+
    scale_x_discrete(na.translate = F)+
    scale_y_continuous(trans = "log10", labels = scales::number_format(accuracy = 1))+
    theme_pubr()+
    theme(text = element_text(size = 20),
          axis.text.x = element_text(angle = 45, vjust = 0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          axis.ticks = element_line(color = "black"),
          axis.ticks.length = unit(0.25, "cm"),
          legend.text = element_markdown())+
    labs(title = title, y = ylab, x = "Days Post Transplant")+
    scale_fill_manual(values=colors)+
    scale_color_manual(values=colors)

  return(list(fig, fig_vio, stat_test, immdata_fig, immdata_sum))
}


num_clones <- function(data, type, metric = "cum_freq", num_cuts = 5, title, ylab = "Allo Frequency", text_size = 20, group_by = c("PatientID", "timepoint", "PTCy")) {
  #' Analyze number of clones and visualize results
  #'
  #' @description This function calculates the number of clones based on specific metrics, performs statistical comparisons between groups, and visualizes the results using boxplots with significance annotations.
  #'
  #' @param data A dataframe containing clone data, including a `SampleID` column.
  #' @param type A character string specifying the grouping variable for the analysis.
  #' @param metric A character string specifying the metric to use for clone frequency calculation (default is "cum_freq").
  #' @param num_cuts An integer indicating the number of bins to divide the timepoints into (default is 5).
  #' @param title A character string for the plot title.
  #' @param ylab A character string for the y-axis label in the plot (default is "Allo Frequency").
  #' @param text_size An integer specifying the size of text in the plot (default is 20).
  #' @param group_by A character vector specifying the grouping variables for clone counting (default is c("PatientID", "timepoint", "PTCy")).
  #' @usage num_clones(data, type, metric, num_cuts, title, ylab, text_size, group_by)
  #' @return A list containing the boxplot with significance annotations, the Wilcoxon test results, and the processed data.

  results <- count_clones(data, type, metric, num_cuts, group_by = group_by)
  data_post <- results[[2]]
  data_post$type_num <- as.numeric(factor(data_post[[type]]))

  # Filter data_post for non-zero clone numbers
  data_post_n0 <- data_post %>%
    filter(num_clones != 0)

  # Perform Wilcoxon tests and adjust p-values
  wilcox_results <- compare_means(as.formula(paste0("num_clones ~ ", type)), data = data_post, group.by = "bins")

  stat_test <- data_post %>%
    group_by(bins) %>%
    pairwise_wilcox_test(as.formula(paste0("num_clones ~ ", type)), p.adjust.method = "hochberg") %>%
    add_xy_position(x = "bins") %>%
    mutate(
      p.adj = p.adjust(p, method = "hochberg"),
      p.adj.signif = symnum(
        p.adj,
        cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1),
        symbols = c("****", "***", "**", "*", "ns")
      )
    )

  print(colnames(data_post)) # Print column names of data_post for debugging

  # Create boxplot with significance annotations and beeswarm overlay
  box_plot_post <- ggboxplot(data_post, x = "bins", y = "num_clones", fill = type) +
    stat_pvalue_manual(stat_test, label = "p.adj.signif", tip.length = 0.01, step.increase = 0.00, bracket.nudge.y = -1, size = 8) +
    geom_beeswarm(data = data_post_n0, position = "dodge", dodge.width = .75, na.rm = TRUE, aes(x = bins, y = num_clones, group = .data[[type]])) +
    scale_y_continuous(trans = 'log10', labels = scales::number_format(accuracy = 0.0001)) +
    theme_pubr() +
    theme(
      text = element_text(size = text_size),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks.length = unit(0.25, "cm"),
      legend.text = element_markdown()
    ) +
    labs(y = ylab, title = title, x = "Days")

  # Prepare data for output
  data_post <- data_post %>%
    mutate(timepoint = as.character(timepoint))
  data <- bind_rows(data_post)

  return(list(box_plot_post, wilcox_results, data)) # Return boxplot, Wilcoxon test results, and processed data
}


avg_clone_fig <- function(immdata, type, title = "", num_cuts = 5, groupings = NA, colors = c("#E6BE60", "#A483AF"), method = "hochberg", legend_title = "", ylab = "") {
  #' Visualize average clone frequency over time
  #'
  #' @description This function processes immunosequencing data to compute average clone frequencies over time, divides them into bins, performs statistical tests, and visualizes the results.
  #'
  #' @param immdata A dataframe containing immunosequencing data with `timepoint`, `PatientID`, and specified `type`.
  #' @param type A character string specifying the grouping variable for the analysis.
  #' @param title A character string for the plot title.
  #' @param num_cuts An integer indicating the number of bins to divide the timepoints into (default is 5).
  #' @param groupings Optional grouping variable for further subsetting (default is NA).
  #' @param colors A character vector specifying colors for the plot (default is c("#E6BE60", "#A483AF")).
  #' @param method A character string specifying the p-value adjustment method (default is "hochberg").
  #' @param legend_title A character string for the legend title.
  #' @param ylab A character string for the y-axis label in the plot.
  #' @usage avg_clone_fig(immdata, type, title, num_cuts, groupings, colors, method, legend_title, ylab)
  #' @return A list containing the plot, statistical test results, processed data, and summary statistics.

  # Convert timepoints to integer and filter data
  immdata <- immdata %>%
    mutate(timepoint = strtoi(timepoint)) %>%
    filter(!is.na(timepoint), timepoint <= 365)

  # Separate early timepoints (<=3 days) and remaining timepoints
  immdata_3 <- immdata %>%
    filter(timepoint <= 3, templates != 0) %>%
    mutate(timepoint = 3, bins = "3")

  immdata <- immdata %>%
    filter(timepoint > 3, templates != 0)

  # Create bins for timepoints
  breaks <- unique(quantile(immdata$timepoint, probs = 0:num_cuts/num_cuts, na.rm = TRUE))
  immdata <- immdata %>%
    mutate(bins = cut(timepoint, breaks = breaks, include.lowest = TRUE))

  # Create an all-timepoint reference dataset
  immdata_all_timepoint <- immdata %>%
    select(all_of(c("timepoint", "PatientID", type))) %>%
    distinct() %>%
    mutate(bins = cut(timepoint, breaks = breaks, include.lowest = TRUE))

  # Combine early and binned timepoints, setting factor levels
  if (nrow(immdata_3) > 0) {
    immdata <- bind_rows(immdata_3, immdata) %>%
      mutate(bins = factor(bins, levels = c("3", levels(immdata_all_timepoint$bins))))
  } else {
    immdata <- immdata %>%
      mutate(bins = factor(bins, levels = levels(immdata_all_timepoint$bins)))
  }

  # Aggregate data for plotting
  immdata_fig <- immdata %>%
    group_by(across(all_of(c("bins", "timepoint", "PatientID", type)))) %>%
    full_join(immdata_all_timepoint, by = c("PatientID", "bins", "timepoint", type)) %>%
    replace_na(list(frequency = 0)) %>%
    group_by(across(all_of(c("bins", "PatientID", type)))) %>%
    summarise(frequency = mean(frequency), .groups = 'drop')

  # Summarize data for error bars
  immdata_sum <- immdata_fig %>%
    group_by(across(all_of(c("bins", type)))) %>%
    summarise(m = mean(frequency), s = sd(frequency), .groups = 'drop')

  # Perform pairwise Wilcoxon tests
  stat_test <- immdata_fig %>%
    group_by(bins) %>%
    filter(n() >= 3) %>%
    pairwise_wilcox_test(as.formula(paste0("frequency ~ ", type)), p.adjust.method = method, exact = FALSE) %>%
    add_xy_position(x = "bins") %>%
    mutate(
      y.position = log10(y.position),
      p.adj = p.adjust(p, method = method),
      p.adj.signif = symnum(
        p.adj,
        cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1),
        symbols = c("****", "***", "**", "*", "ns")
      )
    )

  # Generate the plot
  fig <- ggplot(immdata_sum, aes(x = bins, y = m, fill = .data[[type]], group = .data[[type]], color = .data[[type]])) +
    geom_line() +
    geom_point() +
    geom_ribbon(aes(ymin = pmax(0, m - s), ymax = m + s, linetype = NA), alpha = 0.3) +
    geom_errorbar(aes(ymin = pmax(0, m - s), ymax = m + s), width = 0.1) +
    scale_x_discrete(na.translate = FALSE) +
    theme_minimal() +
    theme(
      text = element_text(size = 16),
      axis.text.x = element_text(angle = 75, vjust = 0.5, size = 12),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      axis.ticks.length = unit(0.25, "cm")
    ) +
    labs(title = title, y = ylab, x = "Days") +
    scale_fill_manual(values = colors) +
    scale_color_manual(values = colors)

  return(list(fig, stat_test, immdata_fig, immdata_sum)) # Return plot, statistical tests, and processed data
}


div_cut <- function(data, type, div_metric = "shannons.entropy", num_cuts = 5, text_size = 20, ylab = "", title = ""){
  #' Analyze diversity metrics and visualize results
  #'
  #' @description This function calculates diversity metrics for different bins of timepoints, performs pairwise Wilcoxon tests, and visualizes the results using boxplots with significance annotations.
  #'
  #' @param data A dataframe containing clone data, including a `SampleID` column.
  #' @param type A character string specifying the grouping variable for the analysis.
  #' @param div_metric A character string specifying the diversity metric to use (default is "shannons.entropy").
  #' @param num_cuts An integer indicating the number of bins to divide the timepoints into (default is 5).
  #' @param text_size An integer specifying the size of text in the plot (default is 20).
  #' @param ylab A character string for the y-axis label in the plot.
  #' @param title A character string for the plot title.
  #' @usage div_cut(data, type, div_metric, num_cuts, text_size, ylab, title)
  #' @return A list containing the boxplot with significance annotations, the Wilcoxon test results, and the processed data.

  results <- div_clones(data, type, div_metric, num_cuts)
  data_pre <- results[[1]]
  data_post <- results[[2]]

  print(unique(data_pre[[type]])) # Print unique values of the grouping variable in pre-stimulation data

  # Summarize post-stimulation data by bins and grouping variable
  data_post_sum <- data_post %>%
    group_by(across(all_of(c("bins", type)))) %>%
    summarise(n_each = n())

  # Perform pairwise Wilcoxon test and add positions for significance annotations
  stat_test <- data_post %>%
    group_by(bins) %>%
    pairwise_wilcox_test(as.formula(paste0(div_metric, " ~ ", type)), p.adjust.method = "hochberg") %>%
    rstatix::add_xy_position(x = "bins")

  # Create boxplot with beeswarm overlay and significance annotations
  box_plot_post <- ggboxplot(data_post, x = "bins", y = div_metric, fill = type, outlier = NA) +
    stat_pvalue_manual(stat_test, label = "p.adj.signif", tip.length = 0.01, step.increase = 0.001) +
    geom_beeswarm(position = "dodge", dodge.width = .75, aes(group = .data[[type]])) +
    scale_y_continuous(labels = scales::number_format(accuracy = 1))+
    theme_pubr() +
    theme(
      legend.text = element_markdown(),
      text = element_text(size = text_size),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(0.25, "cm")
    ) +
    labs(y = ylab, title = title)

  return(list(box_plot_post, stat_test, data_post)) # Return boxplot, statistical test results, and processed post-stimulation data
}


# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================

add_tag <- function(immdata){
  immdata$tag <- paste(immdata$cdr3_amino_acid , immdata$v_resolved , immdata$j_resolved)
  return(immdata)
}


convert_id <- function(immdata){
  ids <- strsplit(immdata$id, "_")
  immdata$timepoints <- sapply(ids, function(x) x[[2]])
  immdata$PatientID <- sapply(ids, function(x) x[[1]])
  return(immdata)
}

div_fig_pat <- function(patient_data, div, type, group, ylim){
  #' Create diversity figures for patient data
  #'
  #' @description This function creates two timeline plots for displaying diversity metrics
  #' over time, split into early (0-365 days) and later (365+ days) time points.
  #'
  #' @param patient_data Dataframe containing patient data
  #' @param div Name of the column containing the diversity metric
  #' @param type Vector of Rep_div values to include
  #' @param group Name of the column to use for grouping and coloring
  #' @param ylim Vector of two numbers specifying y-axis limits
  #' @usage div_fig_pat(patient_data, div, type, group, ylim)
  #' @return list: list(fig_early, fig_later, max_tp)

  selected <- subset(patient_data, patient_data$Rep_div %in% type) %>%
    mutate(SampleID = strtoi(SampleID)) %>%
    filter(!is.na(SampleID), SampleID > 3, is.finite(.data[[div]]))

  selected$GVHD_GRADE_1 <- factor(selected$GVHD_GRADE_1, levels = c('No', 'Severe', 'Mild'))
  max_tp <- max(selected$SampleID)

  fig_early <- ggplot(selected, aes(x = SampleID, y = .data[[div]], group = PatientID, color = .data[[group]], shape = .data[[group]])) +
    geom_point(size = 4) +
    geom_line(size = 1.75) +
    geom_vline(data = selected, aes(xintercept = Onset, color = factor(PatientID)), linetype = 'dashed', show.legend = FALSE) +
    labs(x = "Days Post Transplantation", y = div) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
    scale_colour_discrete(na.translate = F) +
    scale_shape_discrete(na.translate = F) +
    coord_cartesian(xlim = c(0, 365), ylim = ylim, expand = F) +
    theme_pubr() +
    theme(axis.text.x = element_text(size = 30, angle = 75, vjust = 0.5),
          axis.text.y = element_text(size = 30),
          axis.title.x.bottom = element_text(hjust = 0.75),
          axis.line.x.bottom = element_line(linewidth = 1.3),
          axis.line.y = element_line(linewidth = 1.3),
          axis.text = element_text(size = 40))

  fig_later <- ggplot(selected, aes(x = SampleID, y = .data[[div]], group = PatientID, color = .data[[group]], shape = .data[[group]])) +
    geom_point(size = 4) +
    geom_line(size = 1.75) +
    geom_vline(data = selected, aes(xintercept = Onset, color = factor(PatientID)), linetype = 'dashed', show.legend = FALSE) +
    labs(x = "") +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01))+
    scale_colour_discrete(na.translate = F) +
    scale_shape_discrete(na.translate = F) +
    coord_cartesian(xlim = c(365, max_tp), ylim = ylim, expand = F) +
    theme_pubr() +
    theme(axis.text.x = element_text(size = 30, angle = 75, vjust = 0.5),
          axis.text.y = element_text(size = 30),
          axis.line.x.bottom = element_line(linewidth = 1.3),
          axis.title.y = element_blank(),
          axis.text.y.left = element_blank(),
          axis.line.y.left = element_blank(),
          axis.minor.ticks.y.left = element_blank(),
          axis.ticks.y.left = element_blank(),
          axis.text = element_text(size = 40),
          legend.position = "none")

  return(list(fig_early, fig_later, max_tp))
}

plot_metatime_cont <- function(immdata, meta_to_display, time_var, color = "PatientID", labels = c(""), max_tp){
  #' Plot meta timeline
  #'
  #' @description This function creates two timeline plots for displaying metadata over time.
  #' One plot shows early time points (0-365 days) and the other shows later time points.
  #'
  #' @param immdata Dataframe containing the immunosequencing data and metadata
  #' @param meta_to_display Vector of column names in immdata to be displayed as metadata
  #' @param time_var Name of the column in immdata that represents time
  #' @param color Name of the column to be used for coloring points (default: "PatientID")
  #' @param labels Vector of labels for the y-axis (should match length of meta_to_display)
  #' @param max_tp Maximum time point for the second plot
  #' @usage plot_metatime_cont(immdata, meta_to_display, time_var, color, labels, max_tp)
  #' @return list: list(meta_data_plot_early, meta_data_plot_later)

  # Reshape and prepare the data for plotting
  immdata_meta <- immdata %>%
    select(all_of(c(time_var, color, meta_to_display))) %>%
    pivot_longer(cols = meta_to_display, names_to = "Meta") %>%
    mutate(Meta = factor(Meta, levels = rev(meta_to_display)))

  # Create the first plot for early time points (0-365 days)
  meta_data_plot_early <- ggplot(immdata_meta, aes(x = value, y = Meta, color = !!sym(color))) +
    geom_point(shape = 18, size = 9) +
    scale_y_discrete(labels = rev(labels)) +
    theme_pubr() +
    coord_cartesian(xlim = c(0,365)) +
    scale_x_continuous(expand = expansion(mult = c(0,0))) +
    theme(legend.position="none",
          axis.line.x.bottom = element_blank(),
          axis.line.y.left = element_line(size = 1.5),
          axis.title.y.left = element_blank(),
          axis.title.x.bottom = element_blank(),
          axis.minor.ticks.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.x.bottom = element_blank(),
          panel.grid.major.y = element_line(color = "#F0F0F0"))

  # Create the second plot for later time points (366 days to max_tp)
  meta_data_plot_later <- ggplot(immdata_meta, aes(x = value, y = Meta, color = !!sym(color))) +
    geom_point(shape = 18, size = 9) +
    theme_pubr() +
    coord_cartesian(xlim = c(366, max_tp)) +
    scale_x_continuous(expand = expansion(mult = c(0,0))) +
    theme(legend.position="none",
          axis.line.x.bottom = element_blank(),
          axis.title.y.left = element_blank(),
          axis.title.x.bottom = element_blank(),
          axis.text.y.left = element_blank(),
          axis.line.y.left = element_blank(),
          axis.minor.ticks.y.left = element_blank(),
          axis.ticks.y.left = element_blank(),
          axis.minor.ticks.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.x.bottom = element_blank(),
          panel.grid.major.y = element_line(color = "#F0F0F0"))

  # Return both plots as a list
  return(list(meta_data_plot_early, meta_data_plot_later))
}


# ==============================================================================
# END OF FUNCTIONS
# ==============================================================================
