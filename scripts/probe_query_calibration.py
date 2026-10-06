"""Cheap probing experiments on the cached FiQA embeddings from F:\\Dense Retrieval.

Goal: before implementing anything new, measure (a) how much recoverable space exists
beyond the oracle experiment, and (b) which *inference-available* correction signals
actually carry the direction information.

Run with the project venv python:
    & "F:\Dense Retrieval\.venv\Scripts\python.exe" probe_query_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r"F:\Dense Retrieval")
sys.path.insert(0, str(PROJECT / "src"))
from dense_retrieval.datasets import load_beir_dataset  # noqa: E402

pd.set_option("display.width", 240)


def load_case(model_dir: str):
    ds = load_beir_dataset("fiqa", str(PROJECT / "datasets"), "test", False)
    qids = [q.query_id for q in ds.queries]
    dids = list(ds.corpus)
    cache = PROJECT / "cache" / "fiqa" / model_dir
    key = None
    for cand in sorted(p for p in cache.iterdir() if p.is_dir()):
        if (cand / "queries.npy").exists() and (cand / "documents.npy").exists():
            key = cand
            break
    q = np.load(key / "queries.npy").astype(np.float32)
    d = np.load(key / "documents.npy").astype(np.float32)
    assert q.shape[0] == len(qids) and d.shape[0] == len(dids), (q.shape, d.shape, len(qids), len(dids))
    pos_map = {qid: [dids.index(str(x)) for x, r in ds.qrels[qid].items() if int(r) > 0] for qid in qids}
    pos = [pos_map[qid] for qid in qids]
    return qids, dids, q, d, pos


def ranks_for(q: np.ndarray, d: np.ndarray, pos: list[list[int]], topn: int = 60000) -> np.ndarray:
    """Return best relevant rank (1-based, np.inf if not found) per query."""
    scores = q @ d.T
    order = np.argsort(-scores, axis=1)[:, :topn]
    out = np.empty(len(q), dtype=np.float64)
    for i, positives in enumerate(pos):
        rank_of = {int(doc): r + 1 for r, doc in enumerate(order[i])}
        out[i] = min((rank_of.get(p, np.inf) for p in positives), default=np.inf)
    return out


def summarize(name: str, rank: np.ndarray, base_group: np.ndarray, base_rank: np.ndarray):
    hit10 = rank <= 10
    nm = base_group == "near_miss"
    hm = base_group == "hard_miss"
    ht = base_group == "hit"
    return dict(
        variant=name,
        HitRate10=round(100 * hit10.mean(), 2),
        HitRate100=round(100 * (rank <= 100).mean(), 2),
        nearMiss_n=int(nm.sum()),
        nm_into_top10=round(100 * hit10[nm].mean(), 1) if nm.any() else None,
        nm_median_rank_before=round(float(np.median(base_rank[nm])), 1) if nm.any() else None,
        nm_median_rank_after=round(float(np.median(rank[nm])), 1) if nm.any() else None,
        hm_into_top100=round(100 * (rank[hm] <= 100).mean(), 1) if hm.any() else None,
        hit_stay_top10=round(100 * hit10[ht].mean(), 1) if ht.any() else None,
        hit_rank_regression=round(float(np.mean(rank[ht] - base_rank[ht])), 3) if ht.any() else None,
    )


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def ridge_fit_predict(X, Y, folds=5, alpha=1.0):
    """Least-squares (ridge) with k-fold CV; returns out-of-fold predictions."""
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    pred = np.zeros_like(Y)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        Xt = np.hstack([X[train], np.ones((len(train), 1))]).astype(np.float64)
        Xv = np.hstack([X[test], np.ones((len(test), 1))]).astype(np.float64)
        A = Xt.T @ Xt + alpha * np.eye(Xt.shape[1])
        W = np.linalg.solve(A, Xt.T @ Y[train])
        pred[test] = Xv @ W
    return pred


def main() -> None:
    qids, dids, Q, D, pos = load_case("Qwen__Qwen3-Embedding-0.6B")
    print(f"queries={Q.shape} docs={D.shape} dim={Q.shape[1]}")

    base_rank = ranks_for(Q, D, pos)
    base_group = np.where(base_rank <= 10, "hit", np.where(base_rank <= 100, "near_miss", "hard_miss"))
    print("baseline group counts:", pd.Series(base_group).value_counts().to_dict())

    rows = [summarize("v0_baseline", base_rank, base_group, base_rank)]

    mu_d = D.mean(axis=0)
    mu_d = mu_d / np.linalg.norm(mu_d)

    # --- v1: corpus-prior shift, no retrieval feedback at all -------------
    for lam in [0.02, 0.05, 0.1, 0.2, 0.4]:
        q2 = normalize(Q + lam * (mu_d - Q))
        rows.append(summarize(f"v1_corpus_mean_shift_lam{lam}", ranks_for(q2, D, pos), base_group, base_rank))

    # --- v2: classic dense PRF / Rocchio with top-K centroid ---------------
    scores = Q @ D.T
    for K in [5, 10]:
        topk = np.argpartition(-scores, K - 1, axis=1)[:, :K]
        c = D[topk].mean(axis=1)
        for lam in [0.1, 0.3, 0.5]:
            q2 = normalize(Q + lam * (c - Q))
            rows.append(summarize(f"v2_rocchio_K{K}_lam{lam}", ranks_for(q2, D, pos), base_group, base_rank))

    # --- v3/v4: anisotropy / principal-direction removal --------------------
    Dc = D - D.mean(axis=0)
    cov = (Dc.T @ Dc) / len(Dc)
    w, V = np.linalg.eigh(cov)
    top = V[:, np.argsort(-w)]
    for k in [1, 2]:
        U = top[:, :k]
        q2 = normalize(Q - (Q @ U) @ U.T)                      # query side only
        rows.append(summarize(f"v3_query_only_drop_pc{k}", ranks_for(q2, D, pos), base_group, base_rank))
        q3 = normalize(Q - (Q @ U) @ U.T)
        d3 = normalize(D - (D @ U) @ U.T)                      # both sides (needs re-index)
        rows.append(summarize(f"v4_both_drop_pc{k}", ranks_for(q3, d3, pos), base_group, base_rank))

    print("\n### Variant comparison (Qwen3-Embedding-0.6B / FiQA) ###")
    print(pd.DataFrame(rows).to_string(index=False))

    # --- v5: is the correction direction predictable from q alone? ----------
    best_pos = np.array([max(p, key=lambda i: float(Q[j] @ D[i])) for j, p in enumerate(pos)])
    dpos = D[best_pos]
    delta = normalize(dpos - Q)                                  # oracle direction
    print("\n### v5 direction geometry ###")
    align_corpus = (normalize(mu_d - Q) * delta).sum(axis=1)
    align_hit = (normalize(dpos - Q) * normalize(Q)).sum(axis=1)
    for g in ["hit", "near_miss", "hard_miss"]:
        m = base_group == g
        print(f"{g:10s} n={m.sum():4d}  cos(corpus_mean_dir, oracle_dir)={align_corpus[m].mean():.3f}"
              f"  cos(q, oracle_dir)={align_hit[m].mean():.3f}")

    X = normalize(Q)
    for alpha in [1e0, 1e2, 1e4]:
        pred = ridge_fit_predict(X, delta, folds=5, alpha=alpha)
        cos = (normalize(pred) * delta).sum(axis=1)
        nm = base_group == "near_miss"
        print(f"ridge alpha={alpha:>6.0f}: cos(pred,oracle)=all {cos.mean():.3f} | near_miss {cos[nm].mean():.3f} | hit {cos[base_group=='hit'].mean():.3f}")
        for lam in [0.1, 0.2]:
            q2 = normalize(Q + lam * normalize(pred))
            rows2 = summarize(f"v5_ridge_a{alpha:g}_lam{lam}", ranks_for(q2, D, pos), base_group, base_rank)
            print("   ", {k: rows2[k] for k in ["variant", "HitRate10", "nm_into_top10", "hit_stay_top10", "hit_rank_regression"]})

    # oracle direction applied with the same budget, for reference
    for lam in [0.1, 0.2]:
        q2 = normalize(Q + lam * delta)
        rows2 = summarize(f"oracle_dir_lam{lam}", ranks_for(q2, D, pos), base_group, base_rank)
        print("\nREF", {k: rows2[k] for k in ["variant", "HitRate10", "nm_into_top10", "hit_stay_top10", "hit_rank_regression"]})


if __name__ == "__main__":
    main()
