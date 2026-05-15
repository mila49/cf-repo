import scanpy as sc

adata = sc.read_h5ad("adata_vae.h5ad")

# Construir grafo de vecinos usando los embeddings del VAE
sc.pp.neighbors(adata, use_rep="X_vae", n_neighbors=15)

# Clustering Leiden
sc.tl.leiden(adata, resolution=1.0)

# UMAP para visualizar
sc.tl.umap(adata)

# Guardar resultado
adata.write("adata_vae_clustered.h5ad")

# Visualizar
sc.pl.umap(adata, color="leiden")