"""
Visualize UMAP for Best Hyperparameter Configuration
Trains model with specified hyperparameters and generates UMAP plot
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import yaml

from src.pipelines.embedding.dae import DAEPipeline
from src.models import GATRefiner
from src.pipelines.annotations.score_cell_types import load_signatures, get_major_type
from src.interpretability import (
    extract_cluster_attention_matrix,
    plot_attention_heatmap,
    plot_attention_graph,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Best DAE configuration selected by the W&B sweep.
BEST_CONFIG_PATH = (
    project_root
    / "outputs"
    / "dae_sweep"
    / "best_dae_leiden_config.yml"
)
BEST_SUMMARY_PATH = (
    project_root
    / "outputs"
    / "dae_sweep"
    / "best_dae_leiden_summary.yml"
)

OUTPUT_DIR = project_root / "outputs" / "dae_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in YAML file: {path}")

    return data


BEST_CONFIG = load_yaml(BEST_CONFIG_PATH)
BEST_SUMMARY = load_yaml(BEST_SUMMARY_PATH)

# The selected stopping epoch is stored in the summary YAML.
BEST_CONFIG["epochs"] = int(
    BEST_SUMMARY.get(
        "best_epoch",
        BEST_CONFIG.get("epochs", 20),
    )
)

# Reuse DAE embeddings when the DAE configuration is unchanged.
REUSE_DAE_CACHE = True

# Visualization settings
APPLY_GAT_REFINEMENT = False   # refine DAE embeddings with GAT before clustering (z0 → z1)
RUN_GAT_INTERPRETABILITY = True  # train/run GAT on final embeddings for attention analysis only
N_NEIGHBORS_UMAP = 15
MIN_DIST_UMAP = 0.1

# Signature scoring settings
RUN_SIGNATURE_SCORING = True
SIGNATURES_PATH = project_root / "signature_genes_major_types.csv"
SIGNATURE_COLUMN = "major_type"   # column in the CSV that contains signature names
MINIMUM_GENES = 3

# Interpretability settings
RUN_INTERPRETABILITY = True
INTERPRETABILITY_LAYER = "layer1"   # "layer1" (multi-head, averaged) or "layer2" (single head)
INTERPRETABILITY_THRESHOLD = 0.3    # hide edges below this fraction of max attention


def train_and_extract_embeddings(config):
    """Train the DAE or reuse cached embeddings, then return expression objects."""
    print("=" * 80)
    print("TRAINING DAE WITH BEST CONFIGURATION")
    print("=" * 80)
    print(f"Configuration: {config}\n")

    pipeline = DAEPipeline(config_path="embeddings/dae_embedding.yml")
    pipeline.config.update(config)

    # Preprocessing is required to recover obs, HVGs and adata.raw.
    pipeline.setup_data()

    cache_tag = (
        f"genes{int(config['n_top_genes'])}_"
        f"latent{int(config['latent_dim'])}_"
        f"epochs{int(config['epochs'])}"
    )
    embedding_cache = OUTPUT_DIR / f"dae_embeddings_{cache_tag}.npy"
    model_cache = OUTPUT_DIR / f"dae_model_{cache_tag}.pt"

    # adata.raw should contain the full log-normalised expression matrix.
    if pipeline.adata.raw is not None:
        adata_full = pipeline.adata.raw.to_adata()
        print(
            f"Full-gene expression available for signature scoring: "
            f"{adata_full.n_vars} genes"
        )
    else:
        adata_full = pipeline.adata.copy()
        print(
            "Warning: pipeline.adata.raw is None. Signature scoring will only "
            f"use the {adata_full.n_vars} genes present in pipeline.adata."
        )

    if REUSE_DAE_CACHE and embedding_cache.exists():
        embeddings = np.load(embedding_cache)
        expected_shape = (
            pipeline.adata.n_obs,
            int(config["latent_dim"]),
        )

        if embeddings.shape == expected_shape:
            print(f"Reusing cached DAE embeddings: {embedding_cache}")
            return embeddings, pipeline.adata, adata_full

        print(
            "Ignoring incompatible DAE cache: "
            f"found {embeddings.shape}, expected {expected_shape}."
        )

    # Create and train the model only when no compatible cache exists.
    pipeline.setup_model()

    epochs = int(config["epochs"])
    print(f"Training DAE for {epochs} epochs...")

    for epoch in range(epochs):
        pipeline.model.train()
        total_loss = 0.0
        processed = 0

        for batch_x in pipeline.loader:
            batch_x = batch_x.to(pipeline.device)
            loss = pipeline.compute_loss(batch_x)

            pipeline.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            pipeline.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            processed += batch_x.size(0)

        avg_mse = total_loss / max(processed, 1)
        print(f"Epoch {epoch + 1:03d}/{epochs} | MSE: {avg_mse:.4f}")

    print("\nExtracting embeddings...")
    pipeline.model.eval()
    all_embeddings = []

    embedding_loader = DataLoader(
        pipeline.dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for batch_x in embedding_loader:
            batch_x = batch_x.to(pipeline.device)
            latent = pipeline.get_latent_representation(batch_x)
            all_embeddings.append(latent.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(embedding_cache, embeddings)
    torch.save(pipeline.model.state_dict(), model_cache)

    print(f"Saved DAE embeddings: {embedding_cache}")
    print(f"Saved DAE model: {model_cache}")

    return embeddings, pipeline.adata, adata_full


def apply_gat_refinement(embeddings, config):
    """Apply GAT refinement to embeddings."""
    print("\nApplying GAT refinement...")
    
    knn_k = config.get("knn_k", 5)
    gat_hidden_dim = config.get("gat_hidden_dim", 32)
    gat_heads = config.get("gat_heads", 4)
    gat_dropout = config.get("gat_dropout", 0.2)
    gat_epochs = config.get("gat_epochs", 20)
    gat_learning_rate = config.get("gat_learning_rate", 1e-3)
    
    # Build kNN graph
    n_cells = embeddings.shape[0]
    effective_k = max(1, min(knn_k, n_cells - 1))
    
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(embeddings)
    neighbor_idx = nn.kneighbors(return_distance=False)
    
    sources = np.repeat(np.arange(n_cells), effective_k)
    targets = neighbor_idx[:, 1 : effective_k + 1].reshape(-1)
    
    # Symmetrize edges
    src_all = np.concatenate([sources, targets])
    dst_all = np.concatenate([targets, sources])
    
    # Add self-loops
    self_idx = np.arange(n_cells)
    src_all = np.concatenate([src_all, self_idx])
    dst_all = np.concatenate([dst_all, self_idx])
    
    edge_pairs = np.unique(np.stack([src_all, dst_all], axis=1), axis=0)
    edge_index = edge_pairs.T.astype(np.int64)
    
    # Train GAT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = embeddings.shape[1]
    gat_out_dim = input_dim
    
    gat_model = GATRefiner(
        input_dim=input_dim,
        hidden_dim=gat_hidden_dim,
        out_dim=gat_out_dim,
        heads=gat_heads,
        dropout=gat_dropout,
    ).to(device)
    
    x_tensor = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    edge_tensor = torch.from_numpy(edge_index).to(device=device, dtype=torch.long)
    
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=gat_learning_rate)
    gat_loss_fn = torch.nn.MSELoss()
    
    gat_model.train()
    for gat_epoch in range(gat_epochs):
        gat_optimizer.zero_grad()
        _, z0_hat, _ = gat_model(x_tensor, edge_tensor, return_attention=False)
        loss = gat_loss_fn(z0_hat, x_tensor)
        loss.backward()
        gat_optimizer.step()
        
        if (gat_epoch + 1) % 5 == 0:
            print(f"GAT Epoch {gat_epoch + 1}/{gat_epochs} | Loss: {loss.item():.6f}")
    
    # Get refined embeddings
    gat_model.eval()
    with torch.no_grad():
        refined_embeddings, _, _ = gat_model(x_tensor, edge_tensor, return_attention=False)
        refined_embeddings = refined_embeddings.cpu().numpy()

    print(f"Refined embeddings shape: {refined_embeddings.shape}")
    return refined_embeddings, gat_model, edge_index, embeddings


def apply_signature_scoring(adata_source, adata_plot, cluster_key='leiden',
                             signatures_path=None, signature_column=None,
                             minimum_genes=3, random_state=42):
    """Score signature genes on original expression data and copy predictions to adata_plot.

    Parameters
    ----------
    adata_source : AnnData
        Original AnnData with gene expression matrix (n_cells × n_genes).
    adata_plot : AnnData
        AnnData used for UMAP (n_cells × latent_dim) with leiden labels in obs.
    cluster_key : str
        Leiden cluster column in adata_plot.obs.
    signatures_path : str
        Path to signature genes CSV.
    signature_column : str
        Column in the CSV containing signature names.
    minimum_genes : int
        Minimum genes required per signature.
    random_state : int
        Random state for reproducibility.
    """
    if signatures_path is None or not Path(signatures_path).exists():
        print(f"Warning: Signatures file not found: {signatures_path}. Skipping signature scoring.")
        return adata_plot

    print(f"\nRunning signature scoring...")
    print(f"  Signatures file : {signatures_path}")
    print(f"  Signature column: {signature_column}")

    # Work on a copy. When adata_full is available, X contains the full
    # log-normalised expression matrix recovered from pipeline.adata.raw.
    adata_source = adata_source.copy()
    print(f"  Expression genes available: {adata_source.n_vars}")

    # Transfer Leiden labels so signature scores can be averaged per cluster.
    adata_source.obs[cluster_key] = adata_plot.obs[cluster_key].values

    signatures = load_signatures(signatures_path, signature_column)
    print(f"  {len(signatures)} signatures loaded.")

    # Case-insensitive gene mapping
    gene_mapping = {str(g).upper(): str(g) for g in adata_source.var_names}
    used_signatures = {}

    for signature, original_genes in signatures.items():
        available_genes = [
            gene_mapping[g.upper()]
            for g in original_genes
            if g.upper() in gene_mapping
        ]
        available_genes = list(dict.fromkeys(available_genes))
        print(f"  {signature}: {len(available_genes)}/{len(original_genes)} genes available")

        if len(available_genes) < minimum_genes:
            print(f"    Skipping: not enough genes.")
            continue

        sc.tl.score_genes(
            adata_source,
            gene_list=available_genes,
            score_name=f"{signature}_score",
            ctrl_size=min(50, len(available_genes)),
            n_bins=25,
            random_state=random_state,
            use_raw=False,
        )
        used_signatures[signature] = get_major_type(signature)

    if not used_signatures:
        print("Warning: No signature had enough available genes. Skipping signature scoring.")
        return adata_plot

    # Average scores per cluster and assign best signature
    score_columns = [f"{sig}_score" for sig in used_signatures]
    cluster_scores = (
        adata_source.obs
        .groupby(cluster_key, observed=True)[score_columns]
        .mean()
    )
    best_columns = cluster_scores.idxmax(axis=1)
    cluster_scores["assigned_signature"] = best_columns.str.replace("_score", "", regex=False)
    cluster_scores["assigned_major_type"] = cluster_scores["assigned_signature"].map(used_signatures)

    sig_map = {
        str(k): str(v)
        for k, v in cluster_scores["assigned_signature"].astype(str).to_dict().items()
    }
    major_map = {
        str(k): str(v)
        for k, v in cluster_scores["assigned_major_type"].astype(str).to_dict().items()
    }

    cell_clusters = adata_source.obs[cluster_key].astype(str)
    adata_source.obs["predicted_signature"] = cell_clusters.map(sig_map)
    adata_source.obs["predicted_cell_type"] = cell_clusters.map(major_map)

    # Copy predictions back to adata_plot (which holds the UMAP coordinates)
    adata_plot.obs["predicted_signature"] = adata_source.obs["predicted_signature"].values
    adata_plot.obs["predicted_cell_type"] = adata_source.obs["predicted_cell_type"].values

    print("\nSignature scoring complete. Cluster assignments:")
    for cluster in sorted(sig_map.keys(), key=lambda x: int(x) if x.isdigit() else x):
        print(f"  Cluster {cluster}: {sig_map[cluster]} → {major_map.get(cluster, '?')}")

    return adata_plot


def create_umap_plots(embeddings, adata, leiden_resolution=0.5, leiden_n_neighbors=400,
                      output_prefix="best_config",
                      run_signature_scoring=False, signatures_path=None, signature_column=None,
                      config=None, adata_full=None):
    """Create UMAP visualizations.

    Produces two 1×3 figures:
    - ``{prefix}_umap_major.png``: major GT | leiden | major signature scoring
    - ``{prefix}_umap_minor.png``: minor GT (opacity = subtype rank within major) | leiden | signature scoring
    """
    print("\nComputing UMAP...")

    adata_plot = ad.AnnData(X=embeddings)
    adata_plot.obs = adata.obs.copy()

    sc.pp.neighbors(adata_plot, n_neighbors=N_NEIGHBORS_UMAP, use_rep='X')
    sc.tl.umap(adata_plot, min_dist=MIN_DIST_UMAP)

    print(f"Computing Leiden clustering (resolution={leiden_resolution}, n_neighbors={leiden_n_neighbors})...")
    sc.pp.neighbors(adata_plot, n_neighbors=leiden_n_neighbors, use_rep='X')
    sc.tl.leiden(adata_plot, resolution=leiden_resolution, key_added='leiden')

    n_leiden = len(adata_plot.obs['leiden'].unique())
    print(f"Found {n_leiden} Leiden clusters")

    if run_signature_scoring and signatures_path is not None:
        adata_plot = apply_signature_scoring(
            adata_source=adata_full if adata_full is not None else adata,
            adata_plot=adata_plot,
            cluster_key='leiden',
            signatures_path=signatures_path,
            signature_column=signature_column,
            minimum_genes=MINIMUM_GENES,
        )

    print("Creating visualizations...")

    umap_xy   = adata_plot.obsm['X_umap']
    leiden_vals = adata_plot.obs['leiden'].astype(str).values
    ref_major = adata_plot.obs['major'].astype(str).values if 'major' in adata_plot.obs.columns else None
    ref_minor = adata_plot.obs['minor'].astype(str).values if 'minor' in adata_plot.obs.columns else None

    # ── shared colour palette ────────────────────────────────────────────────
    _MAJOR_PALETTE = {
        "Basal":     "#E07B54",
        "Secretory": "#4BACC6",
        "Ciliated":  "#70AD47",
        "Ionocyte":  "#9B59B6",
        "FOXN4+":    "#F1C40F",
        "NE":        "#E74C3C",
    }

    def _hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    # ── helper: scatter with categorical colours ─────────────────────────────
    def _scatter_cat(ax, xy, labels, pal=None, title="", leg_title=""):
        cats = sorted(np.unique(labels), key=str)
        if pal is None:
            raw = sns.color_palette("tab10" if len(cats) <= 10 else "tab20", len(cats))
            pal = dict(zip(cats, [tuple(c) for c in raw]))
        c_arr = [pal.get(lb, (0.5, 0.5, 0.5)) for lb in labels]
        ax.scatter(xy[:, 0], xy[:, 1], c=c_arr, s=1, rasterized=True)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_aspect('equal')
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=pal.get(c, (0.5, 0.5, 0.5)),
                              markersize=8, label=c) for c in cats]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                  frameon=True, fontsize=7, title=leg_title)
        return pal

    # ── helper: ARI / NMI text box ───────────────────────────────────────────
    def _add_metrics(ax, pred, refs):
        lines = []
        for name, ref in refs:
            try:
                ari = adjusted_rand_score(ref, pred)
                nmi = normalized_mutual_info_score(ref, pred)
                lines.append(f"vs {name}:  ARI={ari:.3f}  NMI={nmi:.3f}")
            except Exception as exc:
                lines.append(f"vs {name}: n/a ({exc})")
        if lines:
            props = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray')
            ax.text(0.03, 0.03, '\n'.join(lines), transform=ax.transAxes,
                    fontsize=7.5, verticalalignment='bottom',
                    fontfamily='monospace', bbox=props)

    # ── helper: per-cell RGBA for minor (same hue, opacity = subtype rank) ───
    def _minor_rgba_map(minor_vals, major_vals):
        """Return (per_cell_rgba ndarray [N,4], legend_dict {subtype: rgba})."""
        minor_to_major = {}
        for mn in np.unique(minor_vals):
            parents, counts = np.unique(major_vals[minor_vals == mn], return_counts=True)
            minor_to_major[mn] = parents[counts.argmax()]

        major_to_minors = {}
        for mn, maj in minor_to_major.items():
            major_to_minors.setdefault(maj, []).append(mn)
        for maj in major_to_minors:
            major_to_minors[maj] = sorted(major_to_minors[maj])

        rgba_dict = {}
        for maj, minors in major_to_minors.items():
            base = _hex_to_rgb(_MAJOR_PALETTE.get(maj, "#888888"))
            alphas = np.linspace(0.3, 1.0, max(len(minors), 1))
            for mn, al in zip(minors, alphas):
                rgba_dict[mn] = (*base, float(al))

        per_cell = np.array([rgba_dict[mn] for mn in minor_vals], dtype=np.float32)
        return per_cell, rgba_dict

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1 — Major view (1×3)
    # ════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 8))

    # [0] major GT
    if ref_major is not None:
        _scatter_cat(axes1[0], umap_xy, ref_major,
                     pal={k: _hex_to_rgb(v) for k, v in _MAJOR_PALETTE.items()},
                     title="Major Cell Type\n(ground truth)", leg_title="Major type")
    else:
        axes1[0].set_visible(False)

    # [1] leiden + ARI vs major
    leiden_pal = _scatter_cat(axes1[1], umap_xy, leiden_vals,
                               title=f"Leiden Clustering\n({n_leiden} clusters)",
                               leg_title="Leiden")
    if ref_major is not None:
        _add_metrics(axes1[1], leiden_vals, [("major", ref_major)])

    # [2] predicted major type + ARI vs major
    if 'predicted_cell_type' in adata_plot.obs.columns:
        pred_major = adata_plot.obs['predicted_cell_type'].astype(str).values
        _scatter_cat(axes1[2], umap_xy, pred_major,
                     pal={k: _hex_to_rgb(v) for k, v in _MAJOR_PALETTE.items()},
                     title="Signature Scoring\n(predicted major type)",
                     leg_title="Predicted")
        refs = [("major", ref_major)] if ref_major is not None else []
        _add_metrics(axes1[2], pred_major, refs)
    else:
        axes1[2].set_visible(False)

    _hp = (
        f"lr={config.get('learning_rate')}  epochs={config.get('epochs')}  "
        f"latent={config.get('latent_dim')}  genes={config.get('n_top_genes')}  "
        f"gat_epochs={config.get('gat_epochs')}  knn_k={config.get('knn_k')}  "
        f"gat_heads={config.get('gat_heads')}  gat_hidden={config.get('gat_hidden_dim')}"
    ) if config else ""
    fig1.suptitle(
        f"Major Cell-Type View\n{_hp}",
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    out_major = f"{output_prefix}_umap_major.png"
    fig1.savefig(out_major, dpi=150, bbox_inches='tight')
    print(f"\nSaved UMAP (major) → {out_major}")
    plt.close(fig1)

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2 — Minor view (1×3)
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))

    # [0] minor GT  — same hue as major parent, opacity encodes subtype rank
    if ref_minor is not None and ref_major is not None:
        per_cell_rgba, rgba_dict = _minor_rgba_map(ref_minor, ref_major)
        axes2[0].scatter(umap_xy[:, 0], umap_xy[:, 1],
                         c=per_cell_rgba, s=1, rasterized=True)
        axes2[0].set_title("Minor Cell Type\n(ground truth  |  opacity = subtype rank within major)",
                            fontsize=13, fontweight='bold')
        axes2[0].set_xlabel('UMAP 1', fontsize=11)
        axes2[0].set_ylabel('UMAP 2', fontsize=11)
        axes2[0].set_aspect('equal')
        # Legend sorted by major-parent then alphabetically
        handles_m = []
        for mn in sorted(rgba_dict, key=lambda x: (minor_to_major_str := rgba_dict[x], x)):
            r, g, b, a = rgba_dict[mn]
            handles_m.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=(r, g, b),
                                         alpha=a, markersize=8, label=mn))
        axes2[0].legend(handles=handles_m, bbox_to_anchor=(1.05, 1), loc='upper left',
                        frameon=True, fontsize=7, title="Minor type")
    else:
        axes2[0].set_visible(False)

    # [1] leiden + ARI vs minor
    _scatter_cat(axes2[1], umap_xy, leiden_vals,
                 pal=leiden_pal,
                 title=f"Leiden Clustering\n({n_leiden} clusters)",
                 leg_title="Leiden")
    if ref_minor is not None:
        _add_metrics(axes2[1], leiden_vals, [("minor", ref_minor)])

    # [2] predicted signature (5-class) — same major-palette colours, ARI vs minor
    if 'predicted_signature' in adata_plot.obs.columns:
        pred_sig = adata_plot.obs['predicted_signature'].astype(str).values
        sig_pal = {
            sig: _hex_to_rgb(_MAJOR_PALETTE.get(get_major_type(sig),
                                                  _MAJOR_PALETTE.get(sig, "#888888")))
            for sig in np.unique(pred_sig)
        }
        _scatter_cat(axes2[2], umap_xy, pred_sig,
                     pal=sig_pal,
                     title="Signature Scoring\n(predicted signature type)",
                     leg_title="Predicted sig.")
        refs_sig = [("minor", ref_minor)] if ref_minor is not None else []
        _add_metrics(axes2[2], pred_sig, refs_sig)
    else:
        axes2[2].set_visible(False)

    fig2.suptitle(
        f"Minor Cell-Type View\n{_hp}",
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    out_minor = f"{output_prefix}_umap_minor.png"
    fig2.savefig(out_minor, dpi=150, bbox_inches='tight')
    print(f"Saved UMAP (minor) → {out_minor}")
    plt.close(fig2)

    return adata_plot


def run_attention_interpretability(
    gat_model,
    edge_index: np.ndarray,
    z0: np.ndarray,
    adata_plot,
    layer: str = "layer2",
    threshold: float = 0.3,
    output_prefix: str = "best_config_dae_gat",
) -> None:
    """Extract GAT attention weights and produce heatmap + graph visualizations.

    Must be called after signature scoring so that ``adata_plot.obs`` contains
    both ``'leiden'`` cluster labels and ``'predicted_signature'`` cell-type labels.

    Args:
        gat_model: Trained GATRefiner returned by ``apply_gat_refinement``.
        edge_index: kNN graph used during GAT training, shape ``[2, E]``.
        z0: Pre-GAT embeddings (DAE output), shape ``[N, D]``.
        adata_plot: AnnData with obs columns ``'leiden'`` and optionally
            ``'predicted_signature'`` (populated by signature scoring).
        layer: ``'layer2'`` (default, single-head final) or ``'layer1'``
            (averaged over 4 heads).
        threshold: Edges below ``threshold × max_attention`` are hidden in
            the graph view.
        output_prefix: Filename prefix for the two output PNG files.
    """
    print("\n" + "="*80)
    print("ATTENTION INTERPRETABILITY ANALYSIS")
    print("="*80)

    cluster_labels = adata_plot.obs["leiden"].values
    predicted_signatures = (
        adata_plot.obs["predicted_signature"].values
        if "predicted_signature" in adata_plot.obs.columns
        else None
    )

    # Group by cell-type signature when available so each node in the matrix
    # and graph represents a biological cell type (e.g. Basal, Ciliated) rather
    # than a raw Leiden cluster number.
    group_labels = predicted_signatures if predicted_signatures is not None else cluster_labels

    print(f"\nExtracting attention weights  (layer={layer}) ...")
    attention_matrix, cluster_ids, cluster_sizes, signature_per_cluster = (
        extract_cluster_attention_matrix(
            gat_model=gat_model,
            z0=z0,
            edge_index=edge_index,
            cluster_labels=group_labels,
            predicted_signatures=None,  # cluster_ids are already cell-type names
            layer=layer,
        )
    )

    print(f"  Cell types found : {len(cluster_ids)}  →  {cluster_ids}")
    print(f"  Matrix shape     : {attention_matrix.shape}")

    heatmap_path = f"{output_prefix}_attention_heatmap.png"
    graph_path   = f"{output_prefix}_attention_graph.png"

    print("\nGenerating visualizations...")
    plot_attention_heatmap(
        attention_matrix=attention_matrix,
        cluster_ids=cluster_ids,
        signature_per_cluster=signature_per_cluster,
        output_path=heatmap_path,
    )
    plot_attention_graph(
        attention_matrix=attention_matrix,
        cluster_ids=cluster_ids,
        cluster_sizes=cluster_sizes,
        signature_per_cluster=signature_per_cluster,
        output_path=graph_path,
        threshold=threshold,
    )

    print("\nInterpretability analysis complete.")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("UMAP VISUALIZATION FOR BEST DAE HYPERPARAMETER CONFIGURATION")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in BEST_CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Leiden parameters selected by the DAE sweep.
    leiden_resolution = float(BEST_CONFIG["leiden_resolution"])
    leiden_n_neighbors = int(BEST_CONFIG["n_neighbors"])

    print(f"\nLeiden Clustering Parameters:")
    print(f"  Resolution: {leiden_resolution}")
    print(f"  N Neighbors: {leiden_n_neighbors}")

    # Train and extract embeddings
    embeddings, adata, adata_full = train_and_extract_embeddings(BEST_CONFIG)

    # Apply GAT refinement to embeddings (z0 → z1) before clustering
    if APPLY_GAT_REFINEMENT:
        embeddings, gat_model_refine, edge_index_refine, z0_refine = apply_gat_refinement(embeddings, BEST_CONFIG)
        output_prefix = str(OUTPUT_DIR / "best_config_dae_gat")
    else:
        gat_model_refine, edge_index_refine, z0_refine = None, None, None
        output_prefix = str(OUTPUT_DIR / "best_config_dae")

    # Create UMAP plots (with Leiden clustering + optional signature scoring)
    adata_plot = create_umap_plots(
        embeddings,
        adata,
        leiden_resolution=leiden_resolution,
        leiden_n_neighbors=leiden_n_neighbors,
        output_prefix=output_prefix,
        run_signature_scoring=RUN_SIGNATURE_SCORING,
        signatures_path=SIGNATURES_PATH,
        signature_column=SIGNATURE_COLUMN,
        config=BEST_CONFIG,
        adata_full=adata_full,
    )

    # Save embeddings
    output_h5ad = f"{output_prefix}_embeddings.h5ad"
    ad.settings.allow_write_nullable_strings = True  # required for pd.StringDtype obs columns
    adata_plot.write(output_h5ad)
    print(f"\nSaved embeddings to: {output_h5ad}")

    # GAT interpretability — runs on the final clustered+annotated data.
    # If APPLY_GAT_REFINEMENT is True, reuses the already-trained refinement GAT.
    # If only RUN_GAT_INTERPRETABILITY is True, trains a fresh GAT solely for
    # attention extraction (does not affect clustering or UMAP).
    if RUN_INTERPRETABILITY:
        if APPLY_GAT_REFINEMENT:
            # Reuse the GAT trained for refinement
            gat_model_interp = gat_model_refine
            edge_index_interp = edge_index_refine
            z0_interp = z0_refine
        elif RUN_GAT_INTERPRETABILITY:
            # Train a fresh GAT on the final (unrefinned) embeddings just for attention
            _, gat_model_interp, edge_index_interp, z0_interp = apply_gat_refinement(embeddings, BEST_CONFIG)
        else:
            gat_model_interp = None

        if gat_model_interp is not None:
            run_attention_interpretability(
                gat_model=gat_model_interp,
                edge_index=edge_index_interp,
                z0=z0_interp,
                adata_plot=adata_plot,
                layer=INTERPRETABILITY_LAYER,
                threshold=INTERPRETABILITY_THRESHOLD,
                output_prefix=output_prefix,
            )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()