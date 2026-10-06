from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_one(result_dir: Path):
    q = pd.read_parquet(result_dir / "query_metrics.parquet")
    out_dir = result_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, fn):
        plt.figure(figsize=(7, 4))
        fn()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=160)
        plt.close()

    save("rank_distribution", lambda: (plt.hist(q["best_relevant_rank"].dropna().clip(upper=100), bins=30), plt.xlabel("Best relevant rank (clipped at 100)"), plt.ylabel("Queries")))
    save("positive_similarity", lambda: (q.boxplot(column="best_positive_similarity", by="failure_group"), plt.suptitle(""), plt.title("Positive similarity")))
    save("entropy", lambda: (q.boxplot(column="entropy_top10", by="failure_group"), plt.suptitle(""), plt.title("Top-10 retrieval entropy")))
    save("margin", lambda: (q.boxplot(column="top1_top10_margin", by="failure_group"), plt.suptitle(""), plt.title("Top-1 / Top-10 margin")))
    oracle_path = result_dir / "oracle_correction.parquet"
    if oracle_path.exists():
        o = pd.read_parquet(oracle_path); grouped = o.groupby(["failure_group", "lambda"])[["recall_at_10", "recall_at_20"]].mean().reset_index()
        save("oracle_recall", lambda: (grouped.pivot(index="lambda", columns="failure_group", values="recall_at_10").plot(marker="o"), plt.ylabel("Recall@10"), plt.title("Oracle correction")))
        near = o[o["failure_group"] == "near_miss"]
        if not near.empty:
            save("oracle_rank", lambda: (near.boxplot(column="corrected_rank", by="lambda"), plt.suptitle(""), plt.title("Near-miss corrected ranks")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    for path in Path(args.results).glob("*/*/query_metrics.parquet"):
        plot_one(path.parent)
