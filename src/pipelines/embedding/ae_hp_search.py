"""Joint W&B sweep for AE, GAT, Leiden and signature coherence.

Each run:
1. trains one autoencoder with early stopping;
2. extracts validation embeddings;
3. optionally refines them with a GAT;
4. runs Leiden using the sampled graph parameters;
5. evaluates unsupervised clustering quality;
6. evaluates biological coherence using signature-gene scores;
7. optimises a combined final_score.

The best configuration and summary are saved under outputs/ae_sweep/.
"""

import copy
import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import wandb
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, random_split

from src.models import GATRefiner
from src.pipelines.annotations.score_cell_types import (
    get_major_type,
    load_signatures,
)
from src.pipelines.embedding.ae import AEPipeline
from src.utils import (
    compute_bounded_conductance,
    compute_conductance,
    compute_dbcvi,
    compute_edge_purity,
    compute_local_label_entropy,
    compute_modularity,
    scale_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HP_CONFIG_PATH = PROJECT_ROOT / "config" / "embeddings" / "ae_hp_search.yml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ae_sweep"

with open(HP_CONFIG_PATH, "r", encoding="utf-8") as file:
    HP_CONFIG = yaml.safe_load(file)

SWEEP_SETTINGS = HP_CONFIG.get("sweep", {})
SEARCH_SPACE = HP_CONFIG["search_space"]
SEED = int(HP_CONFIG.get("seed", 42))
VALIDATION_FRACTION = float(HP_CONFIG.get("validation_fraction", 0.20))
SCALE_EMBEDDINGS = bool(HP_CONFIG.get("scale_embeddings", True))
USE_GAT_REFINEMENT = bool(HP_CONFIG.get("use_gat_refinement", True))

TRAINING_CONFIG = HP_CONFIG.get("training", {})
MAX_EPOCHS = int(TRAINING_CONFIG.get("max_epochs", 80))
PATIENCE = int(TRAINING_CONFIG.get("patience", 10))
MIN_DELTA = float(TRAINING_CONFIG.get("min_delta", 1e-4))

METRICS_CONFIG = HP_CONFIG.get("metrics", {})
SILHOUETTE_SAMPLE_SIZE = int(
    METRICS_CONFIG.get("silhouette_sample_size", 10000)
)
COMPUTE_EXPENSIVE_DIAGNOSTICS = bool(
    METRICS_CONFIG.get("compute_expensive_diagnostics", False)
)

OBJECTIVE_WEIGHTS = HP_CONFIG.get(
    "objective_weights",
    {"unsupervised": 0.65, "signatures": 0.35},
)
CLUSTER_CONSTRAINTS = HP_CONFIG.get("cluster_constraints", {})
SIGNATURE_CONFIG = HP_CONFIG.get("signature_scoring", {})
GAT_DEFAULTS = HP_CONFIG.get(
    "gat_defaults",
    {"gat_out_dim": None, "gat_weight_decay": 0.0},
)

# Reused when wandb.agent executes several runs in the same process.
_SIGNATURE_SCORE_CACHE = {}
_SIGNATURE_SOURCE_CACHE = None


def build_sweep_config():
    return {
        "method": SWEEP_SETTINGS.get("method", "bayes"),
        "metric": {
            "name": SWEEP_SETTINGS.get("objective", "composite_score"),
            "goal": "maximize",
        },
        "parameters": {
            parameter: {"values": values}
            for parameter, values in SEARCH_SPACE.items()
        },
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    print("Warning: CUDA is unavailable. Falling back to CPU.")
    return torch.device("cpu")


def resolve_project_path(path_value):
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def train_ae(pipeline, train_loader, val_loader, val_size):
    best_val_mse = float("inf")
    best_train_mse = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        pipeline.model.train()
        total_train_loss = 0.0
        processed_train_samples = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(pipeline.device, non_blocking=True)

            pipeline.optimizer.zero_grad(set_to_none=True)
            loss = pipeline.compute_loss(batch_x)
            loss.backward()
            pipeline.optimizer.step()

            batch_samples = batch_x.size(0)
            total_train_loss += loss.item() * batch_samples
            processed_train_samples += batch_samples

        avg_train_mse = total_train_loss / processed_train_samples

        pipeline.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(pipeline.device, non_blocking=True)
                loss = pipeline.compute_loss(batch_x)
                total_val_loss += loss.item() * batch_x.size(0)

        avg_val_mse = total_val_loss / val_size

        if avg_val_mse < best_val_mse - MIN_DELTA:
            best_val_mse = avg_val_mse
            best_train_mse = avg_train_mse
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(pipeline.model.state_dict())
        else:
            patience_counter += 1

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_mse_epoch": avg_train_mse,
                "val_mse_epoch": avg_val_mse,
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{MAX_EPOCHS} | "
            f"Train MSE: {avg_train_mse:.6f} | "
            f"Val MSE: {avg_val_mse:.6f} | "
            f"Best Val MSE: {best_val_mse:.6f}"
        )

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}"
            )
            break

    if best_model_state is None:
        raise RuntimeError("No valid AE state was saved during training.")

    pipeline.model.load_state_dict(best_model_state)
    return best_train_mse, best_val_mse, best_epoch


def extract_embeddings(pipeline, loader):
    pipeline.model.eval()
    embeddings = []

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(pipeline.device, non_blocking=True)
            latent = pipeline.get_latent_representation(batch_x)
            embeddings.append(latent.cpu().numpy())

    return np.vstack(embeddings)


def build_symmetric_knn_edges(embeddings, knn_k):
    n_cells = embeddings.shape[0]
    effective_k = max(1, min(int(knn_k), n_cells - 1))

    nearest_neighbors = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric="euclidean",
    )
    nearest_neighbors.fit(embeddings)
    neighbor_indices = nearest_neighbors.kneighbors(return_distance=False)

    sources = np.repeat(np.arange(n_cells), effective_k)
    targets = neighbor_indices[:, 1 : effective_k + 1].reshape(-1)

    src_all = np.concatenate([sources, targets, np.arange(n_cells)])
    dst_all = np.concatenate([targets, sources, np.arange(n_cells)])

    edge_pairs = np.unique(
        np.stack([src_all, dst_all], axis=1),
        axis=0,
    )
    return edge_pairs.T.astype(np.int64)


def refine_with_gat(embeddings, pipeline):
    edge_index = build_symmetric_knn_edges(
        embeddings,
        pipeline.config.get("knn_k", 15),
    )

    input_dim = embeddings.shape[1]
    gat_out_dim = GAT_DEFAULTS.get("gat_out_dim")
    if gat_out_dim is None:
        gat_out_dim = input_dim

    gat_model = GATRefiner(
        input_dim=input_dim,
        hidden_dim=int(pipeline.config.get("gat_hidden_dim", 32)),
        out_dim=int(gat_out_dim),
        heads=int(pipeline.config.get("gat_heads", 4)),
        dropout=float(pipeline.config.get("gat_dropout", 0.2)),
    ).to(pipeline.device)

    x_tensor = torch.from_numpy(embeddings).to(
        device=pipeline.device,
        dtype=torch.float32,
    )
    edge_tensor = torch.from_numpy(edge_index).to(
        device=pipeline.device,
        dtype=torch.long,
    )

    optimizer = torch.optim.Adam(
        gat_model.parameters(),
        lr=float(pipeline.config.get("gat_learning_rate", 1e-3)),
        weight_decay=float(GAT_DEFAULTS.get("gat_weight_decay", 0.0)),
    )
    loss_function = torch.nn.MSELoss()
    final_loss = np.nan

    gat_model.train()
    for _ in range(int(pipeline.config.get("gat_epochs", 20))):
        optimizer.zero_grad(set_to_none=True)
        _, reconstructed, _ = gat_model(
            x_tensor,
            edge_tensor,
            return_attention=False,
        )
        loss = loss_function(reconstructed, x_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    gat_model.eval()
    with torch.no_grad():
        refined, _, _ = gat_model(
            x_tensor,
            edge_tensor,
            return_attention=False,
        )
        refined = refined.cpu().numpy()

    del gat_model, x_tensor, edge_tensor
    return refined, final_loss


def run_leiden(embeddings, requested_neighbors, resolution):
    effective_neighbors = max(
        2,
        min(int(requested_neighbors), embeddings.shape[0] - 1),
    )

    adata_temp = ad.AnnData(X=embeddings)
    sc.pp.neighbors(
        adata_temp,
        n_neighbors=effective_neighbors,
        use_rep="X",
        random_state=SEED,
    )
    sc.tl.leiden(
        adata_temp,
        resolution=float(resolution),
        key_added="leiden",
        random_state=SEED,
    )

    labels = pd.Categorical(adata_temp.obs["leiden"]).codes
    return adata_temp, labels, effective_neighbors


def calculate_clustering_metrics(embeddings, adata_temp, cluster_labels):
    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)

    metrics = {
        "silhouette_latent": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "dbcvi": np.nan,
        "edge_purity": np.nan,
        "local_entropy": np.nan,
        "conductance": np.nan,
        "bounded_conductance": np.nan,
        "modularity": np.nan,
    }

    if not 1 < n_clusters < n_samples:
        return metrics

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n_samples)
    metrics["silhouette_latent"] = silhouette_score(
        embeddings,
        cluster_labels,
        sample_size=sample_size if sample_size < n_samples else None,
        random_state=SEED,
    )
    metrics["davies_bouldin"] = davies_bouldin_score(
        embeddings,
        cluster_labels,
    )
    metrics["calinski_harabasz"] = calinski_harabasz_score(
        embeddings,
        cluster_labels,
    )

    if COMPUTE_EXPENSIVE_DIAGNOSTICS:
        try:
            metrics["dbcvi"] = compute_dbcvi(embeddings, cluster_labels)
            adjacency = adata_temp.obsp["connectivities"]
            metrics["edge_purity"] = compute_edge_purity(
                adjacency,
                cluster_labels,
            )
            metrics["local_entropy"] = compute_local_label_entropy(
                adjacency,
                cluster_labels,
            )
            metrics["conductance"] = compute_conductance(
                adjacency,
                cluster_labels,
            )
            metrics["bounded_conductance"] = compute_bounded_conductance(
                adjacency,
                cluster_labels,
            )
            metrics["modularity"] = compute_modularity(
                adjacency,
                cluster_labels,
            )
        except Exception as error:
            print(f"Warning: diagnostic metrics failed: {error}")

    return metrics


def normalise_unsupervised_metrics(metrics):
    silhouette_value = metrics["silhouette_latent"]
    db_value = metrics["davies_bouldin"]
    ch_value = metrics["calinski_harabasz"]

    silhouette_norm = (
        (silhouette_value + 1.0) / 2.0
        if np.isfinite(silhouette_value)
        else 0.0
    )

    davies_bouldin_norm = (
        1.0 / (1.0 + db_value)
        if np.isfinite(db_value)
        else 0.0
    )

    log_ch = (
        np.log1p(ch_value)
        if np.isfinite(ch_value)
        else 0.0
    )

    calinski_harabasz_norm = (
        log_ch / (log_ch + 1.0)
        if log_ch > 0
        else 0.0
    )

    base_quality = (
        0.40 * silhouette_norm
        + 0.35 * davies_bouldin_norm
        + 0.25 * calinski_harabasz_norm
    )

    return {
        "base_quality": float(base_quality),
        "silhouette_norm": float(silhouette_norm),
        "davies_bouldin_norm": float(davies_bouldin_norm),
        "calinski_harabasz_norm": float(
            calinski_harabasz_norm
        ),
    }


def load_signature_source(pipeline):
    global _SIGNATURE_SOURCE_CACHE

    configured_path = resolve_project_path(
        SIGNATURE_CONFIG.get("expression_adata_path")
    )

    if configured_path is not None:
        if _SIGNATURE_SOURCE_CACHE is None:
            if not configured_path.exists():
                raise FileNotFoundError(
                    f"Signature expression AnnData not found: {configured_path}"
                )
            _SIGNATURE_SOURCE_CACHE = sc.read_h5ad(configured_path)
        return _SIGNATURE_SOURCE_CACHE

    if pipeline.adata.raw is not None:
        return pipeline.adata.raw.to_adata()

    print(
        "Warning: signature scoring is using pipeline.adata because "
        "expression_adata_path and adata.raw are unavailable."
    )
    return pipeline.adata


def calculate_signature_scores(pipeline, val_indices):
    if not SIGNATURE_CONFIG.get("enabled", True):
        return None, {}, 0.0

    signatures_path = resolve_project_path(
        SIGNATURE_CONFIG.get("signatures_path")
    )
    signature_column = SIGNATURE_CONFIG.get("signature_column", "major_type")
    minimum_genes = int(SIGNATURE_CONFIG.get("minimum_genes", 3))
    ctrl_size = int(SIGNATURE_CONFIG.get("ctrl_size", 50))
    n_bins = int(SIGNATURE_CONFIG.get("n_bins", 25))

    if signatures_path is None or not signatures_path.exists():
        raise FileNotFoundError(
            f"Signature CSV not found: {signatures_path}"
        )

    cell_names = pipeline.adata.obs_names[np.asarray(val_indices)]

    configured_expression_path = resolve_project_path(
        SIGNATURE_CONFIG.get("expression_adata_path")
    )
    if configured_expression_path is not None:
        source_key = ("file", str(configured_expression_path.resolve()))
    elif pipeline.adata.raw is not None:
        source_key = (
            "raw",
            int(pipeline.adata.raw.n_vars),
            hash(tuple(map(str, pipeline.adata.raw.var_names))),
        )
    else:
        source_key = (
            "pipeline",
            int(pipeline.adata.n_vars),
            hash(tuple(map(str, pipeline.adata.var_names))),
        )

    cache_key = (tuple(cell_names.astype(str)), source_key)

    if cache_key in _SIGNATURE_SCORE_CACHE:
        cached = _SIGNATURE_SCORE_CACHE[cache_key]
        return cached["scores"], cached["major_map"], cached["coverage"]

    expression_adata = load_signature_source(pipeline)
    missing_cells = cell_names.difference(expression_adata.obs_names)
    if len(missing_cells) > 0:
        raise ValueError(
            f"Signature expression data is missing {len(missing_cells)} "
            "validation cells. Cell identifiers must match."
        )

    expression_adata = expression_adata[cell_names].copy()
    signatures = load_signatures(signatures_path, signature_column)
    gene_mapping = {
        str(gene).upper(): str(gene)
        for gene in expression_adata.var_names
    }

    score_columns = {}
    major_map = {}

    for index, (signature, genes) in enumerate(signatures.items()):
        available_genes = [
            gene_mapping[str(gene).upper()]
            for gene in genes
            if str(gene).upper() in gene_mapping
        ]
        available_genes = list(dict.fromkeys(available_genes))

        if len(available_genes) < minimum_genes:
            continue

        internal_column = f"signature_score_{index}"
        sc.tl.score_genes(
            expression_adata,
            gene_list=available_genes,
            score_name=internal_column,
            ctrl_size=min(ctrl_size, len(available_genes)),
            n_bins=n_bins,
            random_state=SEED,
            use_raw=False,
        )
        score_columns[signature] = internal_column
        major_map[signature] = get_major_type(signature)

    if len(score_columns) < 2:
        raise RuntimeError(
            "Fewer than two valid signatures could be scored. Check gene "
            "names, preprocessing and signature_scoring.expression_adata_path."
        )

    signature_scores = pd.DataFrame(
        {
            signature: expression_adata.obs[column].astype(float).values
            for signature, column in score_columns.items()
        },
        index=cell_names,
    )

    coverage = len(score_columns) / max(len(signatures), 1)
    _SIGNATURE_SCORE_CACHE[cache_key] = {
        "scores": signature_scores,
        "major_map": major_map,
        "coverage": coverage,
    }

    print(
        f"Signature scoring: {len(score_columns)}/{len(signatures)} "
        f"signatures available (coverage={coverage:.3f})."
    )
    return signature_scores, major_map, coverage


def evaluate_signature_coherence(signature_scores, cluster_labels, coverage):
    if signature_scores is None:
        return {
            "signature_score": 0.0,
            "signature_confidence": 0.0,
            "signature_cell_agreement": 0.0,
            "signature_coverage": 0.0,
            "assigned_signature_count": 0,
        }

    values = signature_scores.to_numpy(dtype=float)
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    z_scores = (values - means) / stds

    score_frame = pd.DataFrame(
        z_scores,
        columns=signature_scores.columns,
    )
    score_frame["cluster"] = cluster_labels
    cluster_means = score_frame.groupby("cluster", observed=True).mean()

    raw_cluster_scores = cluster_means.to_numpy(dtype=float)
    shifted = raw_cluster_scores - raw_cluster_scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    top_indices = probabilities.argmax(axis=1)
    top_probabilities = probabilities[np.arange(len(top_indices)), top_indices]
    random_baseline = 1.0 / probabilities.shape[1]
    confidence_by_cluster = np.clip(
        (top_probabilities - random_baseline) / (1.0 - random_baseline),
        0.0,
        1.0,
    )

    cluster_ids = cluster_means.index.to_numpy()
    assigned_index_by_cluster = {
        cluster_id: top_index
        for cluster_id, top_index in zip(cluster_ids, top_indices)
    }

    cell_best_indices = z_scores.argmax(axis=1)
    assigned_cell_indices = np.array(
        [assigned_index_by_cluster[label] for label in cluster_labels]
    )
    cell_agreement = float(
        np.mean(cell_best_indices == assigned_cell_indices)
    )

    cluster_counts = pd.Series(cluster_labels).value_counts().reindex(cluster_ids)
    cluster_weights = cluster_counts.to_numpy(dtype=float)
    cluster_weights /= cluster_weights.sum()
    confidence = float(np.sum(confidence_by_cluster * cluster_weights))

    signature_score = float(
        coverage * (0.50 * confidence + 0.50 * cell_agreement)
    )

    return {
        "signature_score": signature_score,
        "signature_confidence": confidence,
        "signature_cell_agreement": cell_agreement,
        "signature_coverage": float(coverage),
        "assigned_signature_count": int(len(np.unique(top_indices))),
    }


def calculate_composite_score(base_quality, cluster_counts):
    n_clusters = len(cluster_counts)

    min_cluster_size = int(cluster_counts.min())
    max_cluster_size = int(cluster_counts.max())

    cluster_size_ratio = (
        max_cluster_size / min_cluster_size
        if min_cluster_size > 0
        else np.inf
    )

    optimal_clusters = 10

    cluster_resolution_score = float(
        np.exp(
            -((n_clusters - optimal_clusters) ** 2)
            / (2 * 5**2)
        )
    )

    cluster_balance_score = (
        1.0 / (1.0 + np.log1p(cluster_size_ratio - 1.0))
        if cluster_size_ratio > 0
        else 0.01
    )

    composite_score = (
        base_quality
        * cluster_resolution_score
        * cluster_balance_score
    )

    return {
        "composite_score": float(composite_score),
        "cluster_resolution_score": float(
            cluster_resolution_score
        ),
        "cluster_balance_score": float(
            cluster_balance_score
        ),
        "cluster_size_ratio": float(cluster_size_ratio),
    }

def calculate_reference_metrics(pipeline, val_indices, cluster_labels):
    metrics = {}

    for column in ("major", "minor"):
        ari_value = np.nan
        nmi_value = np.nan

        if column in pipeline.adata.obs.columns:
            reference = pipeline.adata.obs[column].values[np.asarray(val_indices)]
            try:
                ari_value = adjusted_rand_score(reference, cluster_labels)
                nmi_value = normalized_mutual_info_score(
                    reference,
                    cluster_labels,
                )
            except Exception as error:
                print(f"Warning: ARI/NMI failed for {column}: {error}")

        metrics[f"ari_reference_{column}_val"] = ari_value
        metrics[f"nmi_reference_{column}_val"] = nmi_value

    return metrics


def sweep_worker():
    load_dotenv(PROJECT_ROOT / ".env")
    run = wandb.init()

    try:
        set_seed(SEED)

        pipeline = AEPipeline(config_path="embeddings/ae_embedding.yml")
        pipeline.config.update(dict(wandb.config))
        pipeline.config["device"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        pipeline.device = resolve_device()
        pipeline.setup_data()

        dataset_size = len(pipeline.dataset)
        val_size = max(3, int(VALIDATION_FRACTION * dataset_size))
        if val_size >= dataset_size:
            raise ValueError("Dataset is too small for the validation split.")
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            pipeline.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED),
        )

        pipeline.setup_model()
        batch_size = int(pipeline.config["batch_size"])
        drop_last = train_size % batch_size == 1

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(SEED),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        best_train_mse, best_val_mse, best_epoch = train_ae(
            pipeline,
            train_loader,
            val_loader,
            val_size,
        )

        val_embeddings = extract_embeddings(pipeline, val_loader)
        embeddings_for_clustering = val_embeddings
        gat_loss = np.nan

        if USE_GAT_REFINEMENT:
            embeddings_for_clustering, gat_loss = refine_with_gat(
                val_embeddings,
                pipeline,
            )

        if SCALE_EMBEDDINGS:
            embeddings_for_clustering, _ = scale_embeddings(
                embeddings_for_clustering,
                fit=True,
            )

        adata_temp, cluster_labels, effective_neighbors = run_leiden(
            embeddings_for_clustering,
            pipeline.config["n_neighbors"],
            pipeline.config["leiden_resolution"],
        )

        unique_clusters, cluster_counts = np.unique(
            cluster_labels,
            return_counts=True,
        )
        cluster_counts_series = pd.Series(
            cluster_counts,
            index=unique_clusters,
        )

        clustering_metrics = calculate_clustering_metrics(
            embeddings_for_clustering,
            adata_temp,
            cluster_labels,
        )
        unsupervised_metrics = normalise_unsupervised_metrics(
            clustering_metrics
        )

        signature_scores, major_map, coverage = calculate_signature_scores(
            pipeline,
            val_dataset.indices,
        )
        signature_metrics = evaluate_signature_coherence(
            signature_scores,
            cluster_labels,
            coverage,
        )
        composite_metrics = calculate_composite_score( unsupervised_metrics["base_quality"],
                                                      cluster_counts_series)
        reference_metrics = calculate_reference_metrics(
            pipeline,
            val_dataset.indices,
            cluster_labels,
        )

        final_metrics = {
            "train_mse": best_train_mse,
            "val_mse": best_val_mse,
            "best_epoch": best_epoch,
            **unsupervised_metrics,
            **signature_metrics,
            **composite_metrics,
            **clustering_metrics,
            "n_clusters": int(len(cluster_counts)),
            "min_cluster_size": int(cluster_counts.min()),
            "max_cluster_size": int(cluster_counts.max()),
            "median_cluster_size": float(np.median(cluster_counts)),
            "leiden_resolution": float(
                pipeline.config["leiden_resolution"]
            ),
            "n_neighbors": int(effective_neighbors),
            "gat_reconstruction_loss": gat_loss,
            "use_gat_refinement": USE_GAT_REFINEMENT,
            "scale_embeddings": SCALE_EMBEDDINGS,
            "available_major_types": int(len(set(major_map.values()))),
            "device": str(pipeline.device),
            **reference_metrics,
        }

        wandb.log(final_metrics)

        print("\nRun summary")
        print(f"  Best epoch:             {best_epoch}")
        print(f"  Validation MSE:         {best_val_mse:.6f}")
        print(f"  Clusters:               {len(cluster_counts)}")
        print(
            f"  Unsupervised quality:   "
            f"{unsupervised_metrics['base_quality']:.6f}"
        )
        print(
            f"  Signature score:        "
            f"{signature_metrics['signature_score']:.6f}"
        )
        print(f"  Composite score:        {composite_metrics['composite_score']:.6f}")

    finally:
        run.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_best_run(best_run):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = OUTPUT_DIR / "best_ae_gat_leiden_config.yml"
    summary_path = OUTPUT_DIR / "best_ae_gat_leiden_summary.yml"

    best_config = to_builtin(dict(best_run.config))
    summary_keys = [
        "composite_score",
        "base_quality",
        "cluster_resolution_score",
        "cluster_balance_score",
        "cluster_size_ratio",
        "signature_score",
        "signature_confidence",
        "signature_cell_agreement",
        "signature_coverage",
        "train_mse",
        "val_mse",
        "best_epoch",
        "n_clusters",
        "min_cluster_size",
        "max_cluster_size",
        "silhouette_latent",
        "davies_bouldin",
        "calinski_harabasz",
        "cluster_count_score",
        "small_cluster_score",
        "cluster_balance",
        "ari_reference_major_val",
        "nmi_reference_major_val",
        "ari_reference_minor_val",
        "nmi_reference_minor_val",
    ]
    best_summary = {
        key: to_builtin(best_run.summary.get(key))
        for key in summary_keys
        if best_run.summary.get(key) is not None
    }

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(best_config, file, sort_keys=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(best_summary, file, sort_keys=False)

    return config_path, summary_path


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    if not wandb_project:
        raise ValueError("WANDB_PROJECT is missing from the .env file.")

    sweep_config = build_sweep_config()
    run_count = int(SWEEP_SETTINGS.get("count", 150))

    print(f"W&B project: {wandb_project}")
    print(f"W&B entity:  {wandb_entity}")
    print(f"Sweep method: {sweep_config['method']}")
    print(f"Runs requested: {run_count}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nSearch space:")
    for parameter, values in SEARCH_SPACE.items():
        print(f"  {parameter}: {values}")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=wandb_project,
        entity=wandb_entity,
    )
    wandb.agent(
        sweep_id,
        function=sweep_worker,
        count=run_count,
        project=wandb_project,
        entity=wandb_entity,
    )

    api = wandb.Api()
    resolved_entity = wandb_entity or api.default_entity
    sweep = api.sweep(f"{resolved_entity}/{wandb_project}/{sweep_id}")
    best_run = sweep.best_run()
    config_path, summary_path = save_best_run(best_run)

    print("\n" + "=" * 72)
    print(f"BEST RUN: {best_run.name} ({best_run.id})")
    print("=" * 72)
    print(
    f"Composite score: "
    f"{best_run.summary.get('composite_score', np.nan):.6f}")
    print(
    f"Base quality: "
    f"{best_run.summary.get('base_quality', np.nan):.6f}" )
    print(
        f"Signature score:      "
        f"{best_run.summary.get('signature_score', np.nan):.6f}"
    )
    print(f"N clusters:           {best_run.summary.get('n_clusters', 'N/A')}")
    print(f"Best epoch:           {best_run.summary.get('best_epoch', 'N/A')}")

    print("\nBest hyperparameters:")
    for parameter, value in best_run.config.items():
        print(f"  {parameter}: {value}")

    print(f"\nSaved configuration: {config_path}")
    print(f"Saved summary:       {summary_path}")


if __name__ == "__main__":
    main()