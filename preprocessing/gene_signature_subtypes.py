from pathlib import Path

import pandas as pd


INPUT_FILE = Path("./Supplementary_Table_2.xlsx")
SHEET_NAME = "DEG_All_Subsets"

OUTPUT_SUBTYPES = Path("./signature_genes_subtypes.csv")
OUTPUT_MAJOR_TYPES = Path("./signature_genes_major_types.csv")


SUBTYPE_ORDER = [
    "Basal1",
    "Basal2",
    "Basal3",
    "Basal4",
    "Basal5",
    "Secretory1",
    "Secretory2",
    "Secretory3",
    "Secretory4",
    "Secretory5",
    "Ciliated1",
    "Ciliated2",
    "Ciliated3",
    "FOXN4+",
    "Ionocyte",
    "NE",
]


MAJOR_TYPE_ORDER = [
    "Basal",
    "Secretory",
    "Ciliated",
    "FOXN4+",
    "Ionocyte",
    "NE",
]


def get_major_type(subtype):

    if subtype.startswith("Basal"):
        return "Basal"

    if subtype.startswith("Secretory"):
        return "Secretory"

    if subtype.startswith("Ciliated"):
        return "Ciliated"
    
    if subtype.startswith("FOXN4+"):
        return "FOXN4+"
    
    if subtype.startswith("Ionocyte"):
        return "FOXN4+"
    
    if subtype.startswith("NE"):
        return "NE"

    return subtype


def put_genes_in_same_row(data, group_column, group_order):

    rows = []

    for group_name in group_order:

        group_data = data[
            data[group_column] == group_name
        ]

        genes = (
            group_data["gene"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not genes:
            continue

        row = {
            group_column: group_name,
            "n_genes": len(genes),
        }

        for index, gene in enumerate(genes, start=1):
            row[f"gene_{index}"] = gene

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    # Read the DEG_All_Subsets sheet
    deg_data = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
    )

    # Check that the required columns exist
    required_columns = {
        "cluster",
        "gene",
        "Heatmap",
    }

    missing_columns = required_columns - set(deg_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only genes marked as part of the signature
    signature_data = deg_data[
        deg_data["Heatmap"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    ][["cluster", "gene"]].copy()

    # Remove rows without subtype or gene
    signature_data = signature_data.dropna(
        subset=["cluster", "gene"]
    )

    # Clean text
    signature_data["cluster"] = (
        signature_data["cluster"]
        .astype(str)
        .str.strip()
    )

    signature_data["gene"] = (
        signature_data["gene"]
        .astype(str)
        .str.strip()
    )

    # Remove repeated genes inside the same subtype
    signature_data = signature_data.drop_duplicates(
        subset=["cluster", "gene"]
    )

    # -------------------------------------------------
    # CSV 1: genes grouped by subtype
    # -------------------------------------------------

    subtype_data = signature_data.rename(
        columns={"cluster": "subtype"}
    )

    subtype_csv = put_genes_in_same_row(
        data=subtype_data,
        group_column="subtype",
        group_order=SUBTYPE_ORDER,
    )

    subtype_csv.to_csv(
        OUTPUT_SUBTYPES,
        index=False,
    )

    # -------------------------------------------------
    # CSV 2: genes grouped by major cell type
    # -------------------------------------------------

    signature_data["major_type"] = (
        signature_data["cluster"]
        .apply(get_major_type)
    )

    major_type_csv = put_genes_in_same_row(
        data=signature_data,
        group_column="major_type",
        group_order=MAJOR_TYPE_ORDER,
    )

    major_type_csv.to_csv(
        OUTPUT_MAJOR_TYPES,
        index=False,
    )

    print("CSV files created successfully:")
    print(f"  - {OUTPUT_SUBTYPES}")
    print(f"  - {OUTPUT_MAJOR_TYPES}")

    print("\nGenes per subtype:")
    print(
        subtype_csv[
            ["subtype", "n_genes"]
        ].to_string(index=False)
    )

    print("\nGenes per major cell type:")
    print(
        major_type_csv[
            ["major_type", "n_genes"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()