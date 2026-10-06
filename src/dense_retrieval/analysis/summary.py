from pathlib import Path
import json
import pandas as pd
from scipy.stats import mannwhitneyu

def summarize_results(results_root: str | Path = "results", output: str | Path | None = None) -> dict:
    root = Path(results_root); rows = []; geometry = []
    for metrics_path in root.glob("*/*/metrics.json"):
        dataset, model = metrics_path.parent.parent.name, metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")); rows.append({"dataset": dataset, "model": model, **metrics})
        qpath = metrics_path.parent / "query_metrics.parquet"
        if qpath.exists():
            frame = pd.read_parquet(qpath); grouped = frame.groupby("failure_group")[["top1_score", "best_positive_similarity", "top1_top10_margin", "entropy_top10", "local_distance_10"]].agg(["mean", "median"]).reset_index(); grouped.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else str(c) for c in grouped.columns]; grouped.insert(0, "model", model); grouped.insert(0, "dataset", dataset); geometry.append(grouped)
    comparisons = []
    for metrics_path in root.glob("*/*/metrics.json"):
        dataset, model = metrics_path.parent.parent.name, metrics_path.parent.name; qpath = metrics_path.parent / "query_metrics.parquet"
        if not qpath.exists(): continue
        frame = pd.read_parquet(qpath)
        for metric in ["top1_score", "best_positive_similarity", "top1_top10_margin", "entropy_top10", "local_distance_10"]:
            for a, b in [("hit", "near_miss"), ("hit", "hard_miss"), ("near_miss", "hard_miss")]:
                x = frame.loc[frame.failure_group == a, metric].dropna(); y = frame.loc[frame.failure_group == b, metric].dropna()
                if len(x) and len(y):
                    stat = mannwhitneyu(x, y, alternative="two-sided")
                    comparisons.append({"dataset": dataset, "model": model, "metric": metric, "group_a": a, "group_b": b, "u_statistic": float(stat.statistic), "p_value": float(stat.pvalue), "n_a": len(x), "n_b": len(y)})
    result = {"retrieval_quality": rows, "geometry_by_failure_group": pd.concat(geometry, ignore_index=True).to_dict(orient="records") if geometry else [], "mann_whitney": comparisons}
    if output:
        Path(output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
