import argparse
from src.pipelines.annotations.score_cell_types import run_scoring


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
)

args = parser.parse_args()

run_scoring(args.config)