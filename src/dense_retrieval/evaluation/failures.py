import pandas as pd


def add_failure_groups(query_metrics: pd.DataFrame) -> pd.DataFrame:
    result = query_metrics.copy()

    def classify(rank):
        if pd.notna(rank) and rank <= 10:
            return "hit"
        if pd.notna(rank) and rank <= 100:
            return "near_miss"
        return "hard_miss"

    result["failure_group"] = result["best_relevant_rank"].map(classify)
    return result

