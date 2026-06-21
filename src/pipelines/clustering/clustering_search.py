import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score,silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def compute_clustering_metrics(embeddings, labels):
    """
    Compute internal clustering metrics.
    """
    labels = np.asarray(labels)

    unique_clusters, cluster_sizes = np.unique(
        labels,
        return_counts=True,
    )

    n_clusters = len(unique_clusters)
    n_samples = len(labels)

    metrics = {
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "n_clusters": n_clusters,
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
    }

    # Metrics require at least two clusters and fewer
    # clusters than samples.
    if 1 < n_clusters < n_samples:
        metrics["silhouette"] = silhouette_score(
            embeddings,
            labels,
        )

        metrics["davies_bouldin"] = davies_bouldin_score(
            embeddings,
            labels,
        )

        metrics["calinski_harabasz"] = (
            calinski_harabasz_score(
                embeddings,
                labels,
            )
        )

    return metrics


def create_result(method, parameters, embeddings, labels):
    """
    Create the result dictionary for one clustering configuration.
    """
    metrics = compute_clustering_metrics(
        embeddings,
        labels,
    )

    return {
        "method": method,
        **parameters,
        **metrics,
    }


def run_leiden_search(embeddings,n_neighbors_values,resolution_values,random_state=42):
    """
    Test different Leiden configurations.
    """
    results = []
    n_samples = embeddings.shape[0]

    for n_neighbors in n_neighbors_values:
        effective_neighbors = min(
            n_neighbors,
            n_samples - 1,
        )

        if effective_neighbors < 2:
            continue

        adata = ad.AnnData(X=embeddings)

        sc.pp.neighbors(
            adata,
            n_neighbors=effective_neighbors,
            use_rep="X",
        )

        for resolution in resolution_values:
            cluster_key = (
                f"leiden_n{effective_neighbors}"
                f"_r{resolution}"
            )

            sc.tl.leiden(
                adata,
                resolution=resolution,
                random_state=random_state,
                key_added=cluster_key,
            )

            labels = (
                adata.obs[cluster_key]
                .astype(str)
                .to_numpy()
            )

            result = create_result(
                method="leiden",
                parameters={
                    "n_neighbors": effective_neighbors,
                    "resolution": resolution,
                    "n_clusters_requested": None,
                    "covariance_type": None,
                },
                embeddings=embeddings,
                labels=labels,
            )

            results.append(result)

    return results


def run_louvain_search(embeddings,n_neighbors_values,resolution_values,random_state=42):
    """
    Test different Louvain configurations.
    """
    results = []
    n_samples = embeddings.shape[0]

    for n_neighbors in n_neighbors_values:
        effective_neighbors = min(
            n_neighbors,
            n_samples - 1,
        )

        if effective_neighbors < 2:
            continue

        adata = ad.AnnData(X=embeddings)

        sc.pp.neighbors(
            adata,
            n_neighbors=effective_neighbors,
            use_rep="X",
        )

        for resolution in resolution_values:
            cluster_key = (
                f"louvain_n{effective_neighbors}"
                f"_r{resolution}"
            )

            sc.tl.louvain(
                adata,
                resolution=resolution,
                random_state=random_state,
                key_added=cluster_key,
            )

            labels = (
                adata.obs[cluster_key]
                .astype(str)
                .to_numpy()
            )

            result = create_result(
                method="louvain",
                parameters={
                    "n_neighbors": effective_neighbors,
                    "resolution": resolution,
                    "n_clusters_requested": None,
                    "covariance_type": None,
                },
                embeddings=embeddings,
                labels=labels,
            )

            results.append(result)

    return results


def run_kmeans_search(embeddings,n_clusters_values,random_state=42):
    """
    Test different K-Means configurations.
    """
    results = []
    n_samples = embeddings.shape[0]

    for n_clusters in n_clusters_values:
        if not 1 < n_clusters < n_samples:
            continue

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

        labels = model.fit_predict(embeddings)

        result = create_result(
            method="kmeans",
            parameters={
                "n_neighbors": None,
                "resolution": None,
                "n_clusters_requested": n_clusters,
                "covariance_type": None,
            },
            embeddings=embeddings,
            labels=labels,
        )

        results.append(result)

    return results


def run_gmm_search(embeddings,n_components_values,covariance_types,random_state=42):
    """
    Test different Gaussian Mixture Model configurations.
    """
    results = []
    n_samples = embeddings.shape[0]

    for n_components in n_components_values:
        if not 1 < n_components < n_samples:
            continue

        for covariance_type in covariance_types:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
                reg_covar=1e-6,
            )

            labels = model.fit_predict(embeddings)

            result = create_result(
                method="gaussian_mixture",
                parameters={
                    "n_neighbors": None,
                    "resolution": None,
                    "n_clusters_requested": n_components,
                    "covariance_type": covariance_type,
                },
                embeddings=embeddings,
                labels=labels,
            )

            results.append(result)

    return results


def run_clustering_search(
    embeddings,
    random_state=42,
    scale_embeddings=True,
    leiden_n_neighbors=None,
    leiden_resolution=None,
    louvain_n_neighbors=None,
    louvain_resolution=None,
    kmeans_n_clusters=None,
    gmm_n_components=None,
    gmm_covariance_types=None,
):
    """
    Run all clustering methods and return all results
    and the configuration with the highest Silhouette score.
    """
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must have shape "
            "(n_samples, latent_dim)."
        )

    if embeddings.shape[0] < 3:
        raise ValueError(
            "At least three validation samples are required."
        )

    if scale_embeddings:
        scaler = StandardScaler()

        clustering_embeddings = scaler.fit_transform(
            embeddings
        )
    else:
        clustering_embeddings = embeddings

    all_results = []

    # Leiden
    all_results.extend(
        run_leiden_search(
            embeddings=clustering_embeddings,
            n_neighbors_values=leiden_n_neighbors or [15, 30, 50],
            resolution_values=leiden_resolution or [0.2, 0.5, 0.8, 1.0],
            random_state=random_state,
        )
    )

    # Louvain
    all_results.extend(
        run_louvain_search(
            embeddings=clustering_embeddings,
            n_neighbors_values=louvain_n_neighbors or [15, 30, 50],
            resolution_values=louvain_resolution or [0.2, 0.5, 0.8, 1.0],
            random_state=random_state,
        )
    )

    # K-Means
    all_results.extend(
        run_kmeans_search(
            embeddings=clustering_embeddings,
            n_clusters_values=kmeans_n_clusters or [3, 5, 8, 10, 15],
            random_state=random_state,
        )
    )

    # Gaussian Mixture Model
    all_results.extend(
        run_gmm_search(
            embeddings=clustering_embeddings,
            n_components_values=gmm_n_components or [3, 5, 8, 10, 15],
            covariance_types=gmm_covariance_types or ["full", "diag"],
            random_state=random_state,
        )
    )

    valid_results = [
        result
        for result in all_results
        if np.isfinite(result["silhouette"])
    ]

    if not valid_results:
        raise RuntimeError(
            "No valid clustering configuration was produced."
        )

    best_result = max(
        valid_results,
        key=lambda result: result["silhouette"],
    )

    return all_results, best_result
