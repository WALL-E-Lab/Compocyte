#!/usr/bin/env Rscript
# Benchmark CHETAH reference preprocessing and classification time/memory.
#
# Usage:
#     Rscript train_chetah.R --subset oelen_e1_500k --data-dir DIR --out-dir DIR
#
# Expects in <data_dir>:
#     train_data.mtx.gz, train_genes.txt, train_cells.txt, train_labels.csv
#     test_data.mtx.gz,  test_genes.txt,  test_cells.txt,  test_labels.csv

.libPaths(c("/home/ritwik24222/R/library", .libPaths()))
library(optparse)

option_list <- list(
  make_option(c("--subset"), type="character", default=NULL,
              help="Subset name", metavar="subset"),
  make_option(c("--data-dir"), type="character", default="data",
              help="Data directory", metavar="DIR"),
  make_option(c("--out-dir"), type="character", default="results",
              help="Output directory", metavar="DIR")
)
opt <- parse_args(OptionParser(option_list=option_list))

stopifnot(!is.null(opt$`subset`))
subset <- opt$`subset`
subdir <- opt$`data-dir`
outdir <- opt$`out-dir`
if (!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)

library(CHETAH)
library(scuttle)
library(Matrix)
library(Rtsne)
library(scran)
library(SingleCellExperiment)
library(jsonlite)

read_mtx <- function(mtx_path, genes_path, cells_path) {
  con <- gzfile(mtx_path, "rt")
  m <- Matrix::readMM(con)
  close(con)
  m <- as(m, "CsparseMatrix")
  rownames(m) <- readLines(genes_path)
  colnames(m) <- readLines(cells_path)
  return(m)
}

# ---------- RSS sampler (replaces Rprof) ----------
start_rss_sampler <- function(path) {
  pid <- Sys.getpid()
  script <- tempfile(fileext=".sh")
  writeLines(c(
    "#!/bin/bash",
    sprintf("while kill -0 %d 2>/dev/null; do", pid),
    sprintf("  printf '%%s %%s\\n' \"$(date +%%s.%%N)\" \"$(awk '/VmRSS/{print $2}' /proc/%d/status)\"", pid),
    "  sleep 0.2",
    "done"
  ), script)
  system(sprintf("bash %s >> %s 2>/dev/null", script, path), wait=FALSE)
}

rss_peak_mb <- function(path, t0, t1) {
  if (!file.exists(path)) return(NA_real_)
  d <- tryCatch(read.table(path), error=function(e) NULL)
  if (is.null(d) || nrow(d) == 0) return(NA_real_)
  d <- d[(d$V1 >= t0) & (d$V1 <= t1), , drop=FALSE]
  if (nrow(d) == 0) return(NA_real_)
  round(max(d$V2) / 1024, 1)
}

get_hardware_info <- function() {
  cpu <- tryCatch(system("awk -F: '/model name/{print $2; exit}' /proc/cpuinfo", intern=TRUE), error=function(e) NA_character_)
  ncpu <- tryCatch(as.integer(system("nproc", intern=TRUE)), error=function(e) NA_integer_)
  ram <- tryCatch(as.numeric(system("awk '/MemTotal/{print $2/1024/1024}' /proc/meminfo", intern=TRUE)), error=function(e) NA_real_)
  gpu <- tryCatch(system("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader", intern=TRUE), error=function(e) NA_character_)
  gpu_parsed <- NA_character_
  if (!is.na(gpu) && length(gpu) > 0) gpu_parsed <- trimws(gpu[1])
  gpu_list <- gpu_parsed
  if (!is.na(gpu_parsed)) {
    parts <- strsplit(gpu_parsed, ",\\s*")[[1]]
    if (length(parts) >= 4) {
      gpu_list <- list(
        model = parts[1],
        driver = parts[2],
        memory_total = parts[3],
        memory_free = parts[4]
      )
    }
  }
  list(
    hostname = Sys.info()[["nodename"]],
    cpu_model = trimws(cpu[1]),
    cpu_count = ncpu,
    ram_gb = round(ram, 2),
    gpu = gpu_list,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
}

Data_train <- read_mtx(
  file.path(subdir, "train_data.mtx.gz"),
  file.path(subdir, "train_genes.txt"),
  file.path(subdir, "train_cells.txt")
)
Labels_train <- read.csv(file.path(subdir, "train_labels.csv"))[,1]
Data_test <- read_mtx(
  file.path(subdir, "test_data.mtx.gz"),
  file.path(subdir, "test_genes.txt"),
  file.path(subdir, "test_cells.txt")
)
Labels_test <- read.csv(file.path(subdir, "test_labels.csv"))[,1]

sce <- SingleCellExperiment(
  list(counts = Data_train),
  colData = data.frame(cell_type1 = Labels_train)
)
sce_test <- SingleCellExperiment(
  list(counts = Data_test),
  colData = data.frame(cell_type1 = Labels_test)
)

rss_log <- tempfile(fileext=".txt")
start_rss_sampler(rss_log)

# --- Training phase: reference preprocessing + build CHETAH reference tree ---
t_train0 <- as.numeric(Sys.time())

sce <- logNormCounts(sce)
counts_sce <- logcounts(sce)
stats <- modelGeneVar(sce)
chosen <- getTopHVGs(stats, n=5000)
sce <- sce[chosen, ]
celltypes_sce <- sce$cell_type1
sce <- SingleCellExperiment(
  assays = list(counts = counts_sce),
  colData = DataFrame(celltypes = celltypes_sce)
)
# Build the reference tree by classifying the reference against itself.
# This constructs the hierarchical classifier used later for the test set.
sce <- CHETAHclassifier(input=sce, ref_cells=sce)

t_train1 <- as.numeric(Sys.time())
train_s <- t_train1 - t_train0
train_mem_mb <- rss_peak_mb(rss_log, t_train0, t_train1)

# --- Inference phase: classify test cells ---
t_inf0 <- as.numeric(Sys.time())

sce_test <- sce_test[chosen, ]
sce_test <- logNormCounts(sce_test)
pca_data <- prcomp(t(logcounts(sce_test)), rank=30)
tsne_data <- Rtsne(pca_data$x[,1:30], pca=FALSE, check_duplicates=FALSE)
reducedDims(sce_test) <- list(PCA=pca_data$x, TSNE=tsne_data$Y)
counts_test <- Matrix(logcounts(sce_test), sparse=TRUE)
tsne_test <- reducedDim(sce_test, "TSNE")[,1:2]
sce_test <- SingleCellExperiment(
  assays = list(counts = counts_test),
  reducedDims = SimpleList(TSNE = tsne_test)
)
sce_test <- CHETAHclassifier(input=sce_test, ref_cells=sce)

t_inf1 <- as.numeric(Sys.time())
inf_s <- t_inf1 - t_inf0
inf_mem_mb <- rss_peak_mb(rss_log, t_inf0, t_inf1)

result <- list(
  method = "CHETAH",
  subset = subset,
  n_train = ncol(Data_train),
  n_test = ncol(Data_test),
  n_genes = nrow(Data_train),
  train_s = train_s,
  inference_s = inf_s,
  rss_train_peak_bytes = round(train_mem_mb * 1024 * 1024),
  rss_inference_peak_bytes = round(inf_mem_mb * 1024 * 1024),
  rss_total_peak_bytes = round(max(train_mem_mb, inf_mem_mb, na.rm=TRUE) * 1024 * 1024),
  hardware = get_hardware_info()
)
out_path <- file.path(outdir, paste0(subset, "_CHETAH.json"))
write_json(result, out_path, pretty=TRUE, auto_unbox=TRUE)
print(result)

# No persistent Rprof file is kept for the resource benchmark.
unlink(rss_log)
