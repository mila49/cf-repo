from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import yaml


SUBTYPE_SHEETS = {
    "Basal Subsets": "Basal",
    "Secretory Subsets": "Secretory",
    "Ciliated Subsets": "Ciliated",
}


def load_config(config_path):
    """Load the YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_gene_column(dataframe):
    """Find the column containing the corrected gene names."""
    if "gene.1" in dataframe.columns:
        return "gene.1"

    if "gene" in dataframe.columns:
        return "gene"

    raise ValueError(
        "No gene column was found in the supplementary table."
    )


def extract_subset_signatures(excel_path):
    """
    Extract Basal, Secretory and Ciliated subtype signatures.

    Only genes marked as 'yes' in the Signature column are used.
    """
    signatures = {}
    signature_to_major_type = {}

    for sheet_name, major_type in SUBTYPE_SHEETS.items():
        dataframe = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
        )

        if "cluster" not in dataframe.columns:
            raise ValueError(
                f"The sheet '{sheet_name}' does not contain "
                "a 'cluster' column."
            )

        if "Signature" not in dataframe.columns:
            raise ValueError(
                f"The sheet '{sheet_name}' does not contain "
                "a 'Signature' column."
            )

        gene_column = get_gene_column(dataframe)

        signature_rows = dataframe[
            dataframe["Signature"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        ].copy()

        signature_rows = signature_rows.dropna(
            subset=["cluster", gene_column]
        )

        for subtype, subtype_data in signature_rows.groupby("cluster"):
            subtype = str(subtype).strip()

            genes = (
                subtype_data[gene_column]
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            signatures[subtype] = genes
            signature_to_major_type[subtype] = major_type

    return signatures, signature_to_major_type


def extract_foxn4_signature(excel_path):
    """
    Extract the FOXN4+ signature from the 'Major Cell Types' sheet.
    """
    dataframe = pd.read_excel(
        excel_path,
        sheet_name="Major Cell Types",
    )

    if "cluster" not in dataframe.columns:
        raise ValueError(
            "The 'Major Cell Types' sheet does not contain "
            "a 'cluster' column."
        )

    gene_column = get_gene_column(dataframe)

    foxn4_rows = dataframe[
        dataframe["cluster"]
        .astype(str)
        .str.upper()
        .str.contains("FOXN4", na=False)
    ].copy()

    if "Signature" in dataframe.columns:
        signature_filter = (
            foxn4_rows["Signature"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        )

        if signature_filter.any():
            foxn4_rows = foxn4_rows[signature_filter]

    foxn4_rows = foxn4_rows.dropna(
        subset=[gene_column]
    )

    genes = (
        foxn4_rows[gene_column]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if not genes:
        raise ValueError(
            "No FOXN4+ signature genes were found in "
            "the 'Major Cell Types' sheet."
        )

    return genes


def extract_all_signatures(excel_path):
    """Extract subtype signatures and add FOXN4+."""
    signatures, signature_to_major_type = (
        extract_subset_signatures(excel_path)
    )

    signatures["FOXN4+"] = extract_foxn4_signature(
        excel_path
    )

    signature_to_major_type["FOXN4+"] = "FOXN4+"

    return signatures, signature_to_major_type


def get_gene_mapping(adata, use_raw):
    """
    Create a case-insensitive mapping between the Excel genes
    and the genes available in AnnData.
    """
    if use_raw:
        if adata.raw is None:
            raise ValueError(
                "use_raw is true, but adata.raw is empty."
            )

        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names

    return {
        str(gene).upper(): str(gene)
        for gene in var_names
    }


def filter_available_genes(signature_genes, gene_mapping):
    """Keep only genes that exist in AnnData."""
    available_genes = []

    for gene in signature_genes:
        matched_gene = gene_mapping.get(
            str(gene).upper()
        )

        if matched_gene is not None:
            available_genes.append(matched_gene)

    return list(dict.fromkeys(available_genes))


def calculate_scores(
    adata,
    signatures,
    use_raw,
    minimum_genes,
    random_state,
):
    """Calculate one signature score per cell."""
    gene_mapping = get_gene_mapping(
        adata,
        use_raw,
    )

    coverage_rows = []

    for signature_name, original_genes in signatures.items():
        available_genes = filter_available_genes(
            original_genes,
            gene_mapping,
        )

        original_count = len(original_genes)
        available_count = len(available_genes)

        if original_count > 0:
            coverage = available_count / original_count
        else:
            coverage = 0

        coverage_rows.append(
            {
                "signature": signature_name,
                "original_genes": original_count,
                "available_genes": available_count,
                "coverage": coverage,
            }
        )

        print(
            f"{signature_name}: "
            f"{available_count}/{original_count} genes available"
        )

        if available_count < minimum_genes:
            print(
                f"Warning: {signature_name} was skipped because "
                f"only {available_count} genes were available."
            )
            continue

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=f"{signature_name}_score",
            ctrl_size=min(50, available_count),
            n_bins=25,
            random_state=random_state,
            use_raw=use_raw,
        )

    return pd.DataFrame(coverage_rows)


def assign_clusters(
    adata,
    cluster_key,
    signature_to_major_type,
):
    """
    Average the scores within each cluster and assign the cluster
    to the signature with the highest average score.
    """
    score_to_signature = {}

    for signature_name in signature_to_major_type:
        score_column = f"{signature_name}_score"

        if score_column in adata.obs.columns:
            score_to_signature[score_column] = signature_name

    score_columns = list(score_to_signature.keys())

    if not score_columns:
        raise ValueError(
            "No signature scores were calculated."
        )

    cluster_scores = (
        adata.obs
        .groupby(cluster_key, observed=True)[score_columns]
        .mean()
    )

    best_score_column = cluster_scores[
        score_columns
    ].idxmax(axis=1)

    cluster_scores["assigned_signature"] = (
        best_score_column.map(score_to_signature)
    )

    cluster_scores["assigned_major_type"] = (
        cluster_scores["assigned_signature"]
        .map(signature_to_major_type)
    )

    sorted_scores = np.sort(
        cluster_scores[score_columns].to_numpy(),
        axis=1,
    )

    cluster_scores["best_score"] = sorted_scores[:, -1]

    if len(score_columns) >= 2:
        cluster_scores["score_margin"] = (
            sorted_scores[:, -1]
            - sorted_scores[:, -2]
        )
    else:
        cluster_scores["score_margin"] = np.nan

    signature_mapping = (
        cluster_scores["assigned_signature"]
        .astype(str)
        .to_dict()
    )

    major_type_mapping = (
        cluster_scores["assigned_major_type"]
        .astype(str)
        .to_dict()
    )

    return (
        cluster_scores,
        signature_mapping,
        major_type_mapping,
    )


def run_scoring(config_path):
    """Run the complete scoring pipeline."""
    config = load_config(config_path)

    adata_path = config["adata_path"]
    supplementary_path = config[
        "supplementary_table_path"
    ]
    cluster_key = config["cluster_key"]

    use_raw = config.get("use_raw", True)
    minimum_genes = config.get("minimum_genes", 5)
    random_state = config.get("random_state", 42)

    output_adata_path = Path(
        config["output_adata_path"]
    )

    output_cluster_scores_path = Path(
        config["output_cluster_scores_path"]
    )

    output_assignments_path = Path(
        config["output_assignments_path"]
    )

    output_coverage_path = Path(
        config["output_signature_coverage_path"]
    )

    output_paths = [
        output_adata_path,
        output_cluster_scores_path,
        output_assignments_path,
        output_coverage_path,
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(f"Loading AnnData from: {adata_path}")

    adata = sc.read_h5ad(adata_path)

    if cluster_key not in adata.obs.columns:
        raise KeyError(
            f"The cluster column '{cluster_key}' was not found. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )

    print("Extracting signatures...")

    signatures, signature_to_major_type = (
        extract_all_signatures(supplementary_path)
    )

    print(
        f"{len(signatures)} signatures were extracted."
    )

    print("Calculating scores...")

    coverage_table = calculate_scores(
        adata,
        signatures,
        use_raw,
        minimum_genes,
        random_state,
    )

    print("Assigning clusters...")

    (
        cluster_scores,
        signature_mapping,
        major_type_mapping,
    ) = assign_clusters(
        adata,
        cluster_key,
        signature_to_major_type,
    )

    cluster_values = adata.obs[
        cluster_key
    ].astype(str)

    signature_mapping = {
        str(cluster): signature
        for cluster, signature in signature_mapping.items()
    }

    major_type_mapping = {
        str(cluster): major_type
        for cluster, major_type in major_type_mapping.items()
    }

    adata.obs["predicted_signature"] = (
        cluster_values.map(signature_mapping)
    )

    adata.obs["predicted_major_cell_type"] = (
        cluster_values.map(major_type_mapping)
    )

    assignments = cluster_scores[
        [
            "assigned_signature",
            "assigned_major_type",
            "best_score",
            "score_margin",
        ]
    ]

    cluster_scores.to_csv(
        output_cluster_scores_path
    )

    assignments.to_csv(
        output_assignments_path
    )

    coverage_table.to_csv(
        output_coverage_path,
        index=False,
    )

    adata.write_h5ad(
        output_adata_path
    )

    print("\nFinal cluster assignments:")
    print(assignments)

    print("\nFiles saved:")
    print(f"- {output_adata_path}")
    print(f"- {output_cluster_scores_path}")
    print(f"- {output_assignments_path}")
    print(f"- {output_coverage_path}")