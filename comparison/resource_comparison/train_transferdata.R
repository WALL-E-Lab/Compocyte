#!/usr/bin/env Rscript
# Benchmark transferData (Seurat) reference integration and label transfer time/memory.
#
# Usage:
#     Rscript train_transferdata.R --subset oelen_e1_500k --data-dir DIR --out-dir DIR
#
# Seurat's v1-style integration (FindIntegrationAnchors + IntegrateData) does
# not scale to the raw sample counts of these datasets:
#   * FindIntegrationAnchors runs FindAnchors on every C(G,2) pair of
#     integration groups, so 550 samples -> 150,975 pairwise fits (this is
#     what made the original run hang for > 34 h).
#   * IntegrateData -> FindIntegrationMatrix materialises an
#     n_anchors x n_anchors matrix, which overflows scipy's int32 index
#     limit ("would have more than 2^31-1 nonzero entries") once
#     n_anchors > ~46341.
# We therefore bucket the samples into a bounded number of balanced
# integration groups and cap k.anchor so that the anchor matrix stays inside
# the int32 limit, and we retry with finer groups if Seurat still complains.

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

library(Seurat)
library(jsonlite)
options(future.globals.maxSize = 8000 * 1024^2)

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

# Number of integration groups such that the anchor matrix of a single anchor
# group stays inside scipy's int32 limit: (k.anchor * cells_per_group)^2 < 2^31
choose_n_groups <- function(n_cells, k_anchor=3, max_anchors=30000) {
  target <- floor(max_anchors / k_anchor)
  max(5, min(64, ceiling(n_cells / target)))
}

# Bucket sample_ids into `k` groups of roughly equal cell count
bucket_samples <- function(sids, k) {
  tab <- table(sids)
  sizes <- as.integer(tab)
  names_sid <- names(tab)
  ord <- order(-sizes)
  buckets <- vector("list", k)
  bsize <- integer(k)
  for (i in ord) {
    b <- which.min(bsize)
    buckets[[b]] <- c(buckets[[b]], names_sid[i])
    bsize[b] <- bsize[b] + sizes[i]
  }
  buckets
}

Data_train <- read_mtx(
  file.path(subdir, "train_data.mtx.gz"),
  file.path(subdir, "train_genes.txt"),
  file.path(subdir, "train_cells.txt")
)
Labels_train <- read.csv(file.path(subdir, "train_labels.csv"))[,1]
Meta_train <- read.csv(file.path(subdir, "train_metadata.csv"))
Data_test <- read_mtx(
  file.path(subdir, "test_data.mtx.gz"),
  file.path(subdir, "test_genes.txt"),
  file.path(subdir, "test_cells.txt")
)
Labels_test <- read.csv(file.path(subdir, "test_labels.csv"))[,1]

ncells <- ncol(Data_train)
k_anchor <- if (ncells <= 100000) 5 else if (ncells <= 250000) 3 else 2
n_groups <- choose_n_groups(ncells, k_anchor=k_anchor)

seuratobj <- CreateSeuratObject(counts = Data_train)
seuratobj$Labels <- Labels_train
seuratobj$sample_id <- Meta_train$sample_id

test_seuratobj <- CreateSeuratObject(counts = Data_test)
test_seuratobj$Labels <- Labels_test

rss_log <- tempfile(fileext=".txt")
start_rss_sampler(rss_log)

# --- Training phase: reference preprocessing and integration ---
t_train0 <- as.numeric(Sys.time())

# Integrate the reference with `n_groups_try` integration groups.  Seurat may
# still merge groups into larger anchor groups, so retry with finer groups if
# the anchor matrix overflows the int32 index limit.
integrate_reference <- function(n_groups_try, k_anchor_try) {
  buckets <- bucket_samples(seuratobj$sample_id, n_groups_try)
  bucket_of <- rep(NA_character_, ncol(seuratobj))
  for (b in seq_along(buckets)) {
    bucket_of[seuratobj$sample_id %in% buckets[[b]]] <- paste0("G", b)
  }
  obj <- seuratobj
  obj$integration_group <- bucket_of

  obj.list <- SplitObject(obj, split.by = "integration_group")
  obj.list <- Filter(function(o) ncol(o) >= 100, obj.list)
  obj.list <- lapply(obj.list, function(o) {
    o <- NormalizeData(o, verbose=FALSE)
    o <- FindVariableFeatures(o, selection.method="vst", nfeatures=2000, verbose=FALSE)
    o
  })
  if (length(obj.list) <= 1) {
    return(obj.list[[1]])
  }
  anchors <- FindIntegrationAnchors(
    object.list = obj.list,
    dims = 1:30,
    k.anchor = k_anchor_try,
    k.filter = 100,
    k.score = 30,
    verbose = FALSE
  )
  integrated <- IntegrateData(anchorset = anchors, dims = 1:30)
  DefaultAssay(integrated) <- "integrated"
  integrated
}

integrated <- NULL
attempt <- 0
groups_try <- n_groups
k_try <- k_anchor
while (is.null(integrated) && attempt < 3) {
  attempt <- attempt + 1
  message(sprintf("transferData: integration attempt %d (n_groups=%d, k.anchor=%d)",
                  attempt, groups_try, k_try))
  res <- tryCatch(
    integrate_reference(groups_try, k_try),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    message(sprintf("transferData: attempt %d failed: %s", attempt, conditionMessage(res)))
    groups_try <- min(64, groups_try * 2)
    k_try <- max(1, k_try - 1)
  } else {
    integrated <- res
    n_groups_used <- groups_try
    k_anchor_used <- k_try
  }
}
if (is.null(integrated)) {
  stop("transferData: integration failed after 3 attempts")
}

seuratobj.integrated <- integrated
seuratobj.integrated <- ScaleData(seuratobj.integrated, verbose=FALSE)
seuratobj.integrated <- RunPCA(seuratobj.integrated, npcs=30, verbose=FALSE)

test_seuratobj <- NormalizeData(test_seuratobj, verbose=FALSE)
test_seuratobj <- FindVariableFeatures(test_seuratobj, selection.method="vst", nfeatures=2000, verbose=FALSE)
test.anchors <- FindTransferAnchors(
  reference=seuratobj.integrated,
  query=test_seuratobj,
  dims=1:30,
  reference.reduction="pca"
)

t_train1 <- as.numeric(Sys.time())
train_s <- t_train1 - t_train0
train_mem_mb <- rss_peak_mb(rss_log, t_train0, t_train1)

# --- Inference phase: TransferData label prediction ---
t_inf0 <- as.numeric(Sys.time())

predictions <- TransferData(
  anchorset=test.anchors,
  refdata=seuratobj.integrated$Labels,
  dims=1:30
)

t_inf1 <- as.numeric(Sys.time())
inf_s <- t_inf1 - t_inf0
inf_mem_mb <- rss_peak_mb(rss_log, t_inf0, t_inf1)

result <- list(
  method = "transferData",
  subset = subset,
  n_train = ncol(Data_train),
  n_test = ncol(Data_test),
  n_genes = nrow(Data_train),
  n_integration_groups = n_groups_used,
  k_anchor = k_anchor_used,
  integration_attempts = attempt,
  train_s = train_s,
  inference_s = inf_s,
  rss_train_peak_bytes = round(train_mem_mb * 1024 * 1024),
  rss_inference_peak_bytes = round(inf_mem_mb * 1024 * 1024),
  rss_total_peak_bytes = round(max(train_mem_mb, inf_mem_mb, na.rm=TRUE) * 1024 * 1024),
  hardware = get_hardware_info()
)
out_path <- file.path(outdir, paste0(subset, "_transferData.json"))
write_json(result, out_path, pretty=TRUE, auto_unbox=TRUE)
print(result)

# No persistent Rprof file is kept for the resource benchmark.
unlink(rss_log)
