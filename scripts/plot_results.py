from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_one(result_dir: Path):
    q = pd.read_parquet(result_dir / "query_metrics.parquet")
    out_dir = result_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, draw):
        figure, axis = plt.subplots(figsize=(7, 4))
        try:
            draw(axis)
            figure.tight_layout()
            figure.savefig(out_dir / f"{name}.png", dpi=160)
        finally:
            plt.close(figure)

    def rank_distribution(axis):
        axis.hist(q["best_relevant_rank"].dropna().clip(upper=100), bins=30)
        axis.set_xlabel("Best relevant rank (clipped at 100)")
        axis.set_ylabel("Queries")

    def grouped_boxplot(axis, column, title):
        q.boxplot(column=column, by="failure_group", ax=axis)
        axis.set_title(title)
        axis.figure.suptitle("")

    save("rank_distribution", rank_distribution)
    save(
        "positive_similarity",
        lambda axis: grouped_boxplot(
            axis, "best_positive_similarity", "Positive similarity"
        ),
    )
    save(
        "entropy",
        lambda axis: grouped_boxplot(axis, "entropy_top10", "Top-10 retrieval entropy"),
    )
    save(
        "margin",
        lambda axis: grouped_boxplot(
            axis, "top1_top10_margin", "Top-1 / Top-10 margin"
        ),
    )
    oracle_path = result_dir / "oracle_correction.parquet"
    if oracle_path.exists():
        o = pd.read_parquet(oracle_path)
        grouped = (
            o.groupby(["failure_group", "lambda"])[
                ["recall_at_10", "recall_at_20"]
            ]
            .mean()
            .reset_index()
        )

        def oracle_recall(axis):
            grouped.pivot(
                index="lambda", columns="failure_group", values="recall_at_10"
            ).plot(marker="o", ax=axis)
            axis.set_ylabel("Recall@10")
            axis.set_title("Oracle correction")

        save("oracle_recall", oracle_recall)
        near = o[o["failure_group"] == "near_miss"]
        if not near.empty:
            def oracle_rank(axis):
                near.boxplot(column="corrected_rank", by="lambda", ax=axis)
                axis.set_title("Near-miss corrected ranks")
                axis.figure.suptitle("")

            save("oracle_rank", oracle_rank)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    for path in Path(args.results).glob("*/*/query_metrics.parquet"):
        plot_one(path.parent)
