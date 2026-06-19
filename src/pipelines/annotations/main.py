from src.pipelines.annotation.score_cell_types import run_scoring

if __name__ == "__main__":
    run_scoring(
        config_path="config/annotation/cell_type_scoring.yml"
    )