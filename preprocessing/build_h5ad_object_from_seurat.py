import scanpy as sc
import pandas as pd
from dotenv import load_dotenv, get_key
from pathlib import Path
from anndata import AnnData

# Load environment variables from .env file
load_dotenv()
root_directory = Path(get_key(".env", "ROOT_DIRECTORY"))
dataset_path = root_directory / "Dataset"

for dir in ["raw", "copper", "silver", "gold"]:
    (dataset_path / dir).mkdir(parents=True, exist_ok=True)

# Read matrix data and transpose it to have cells as rows and genes as columns
adata: AnnData = sc.read_mtx(root_directory / "Dataset" / "raw" / "matrix.mtx").T

# Read gene, metadata and UMAP information
genes = pd.read_csv(root_directory / "Dataset" / "raw" / "genes.csv", header=None)
metadata = pd.read_csv(root_directory / "Dataset" / "raw" / "metadata.csv", index_col=0)
umap = pd.read_csv(root_directory / "Dataset" / "raw" / "umap.csv", index_col=0)

# Save the data as .h5ad for future use
adata.var_names = genes[0].values
adata.obs = metadata
adata.obsm['X_umap'] = umap.values

# Write data to .h5ad file
adata.write(root_directory / "Dataset" / "copper" / "data.h5ad")