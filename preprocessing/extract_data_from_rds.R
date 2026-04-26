# Load Seurat libraries
library(Matrix)
library(Seurat)
library(dotenv)

# Set project directory
env_file_path <- file.path("")
load_dot_env(file = env_file_path)

project_dir <- Sys.getenv("ROOT_DIRECTORY")

# Load data into environment
data <- readRDS(file.path(project_dir, "Dataset", "raw", "GSE150674_Seurat_Object.rds"))

# Extract cell-genes matrix
counts <- GetAssayData(data, assay = "RNA", layer = "counts")

# Write matrix
writeMM(counts, file = file.path(project_dir, "Dataset", "raw", "matrix.mtx"))

# Write genes names
write.table(
  rownames(data),
  file = file.path(project_dir, "Dataset", "raw", "genes.csv"), 
  col.names = FALSE,
  row.names = FALSE,
  quote = FALSE
)

# Write metadata
write.csv(data@meta.data, file = file.path(project_dir, "Dataset", "raw", "metadata.csv"))

# Write PCA
write.csv(Embeddings(data, "pca"), file = file.path(project_dir, "Dataset", "raw", "pca.csv"))

# Write UMAP
write.csv(Embeddings(data, "umap"), file = file.path(project_dir, "Dataset", "raw", "umap.csv"))