# Query representation calibration: research conclusion

## Decision

Under the tested single-vector setting, frozen retrieval is the reliable
deployment policy. Neither query-only nor corpus-aware correction provides a
stable held-out-corpus improvement, so no correction method should replace the
frozen baseline on the present evidence.

The original hypothesis must be split:

1. **Supported:** query-conditioned frozen-corpus geometry contains
   transferable information about a nearest-positive correction direction
   beyond the query embedding alone.
2. **Not supported:** that incremental direction information can be converted
   into stable ranking utility on an unseen corpus.

This is a negative conclusion about transferable correction utility, not a
claim that corpus vectors contain no information.

## Evidence ladder

All primary transfer experiments use strict leave-one-corpus-out evaluation
over FiQA, ArguAna, SciFact, and NFCorpus. The embedding model, document
embeddings, and document index remain frozen. Held-out qrels are excluded from
representation construction, model fitting, and hyperparameter selection.

| Question | Experiment | Result |
| --- | --- | --- |
| Does a useful correction space exist? | Full and low-rank oracle correction; held-out oracle benefit gate | Yes. Oracle corrections produce large gains, and oracle gating has positive mean NDCG headroom for every embedding model. |
| Can query-only features predict the correction? | Ridge prediction in a shared low-rank space | Partially, but predicted corrections retain little of the oracle retrieval gain. |
| Does corpus geometry add direction information conditional on the query? | Matched q-only, fixed-reference, and corpus-local prototype regressors | Yes. Corpus-local geometry improves correction-coordinate NMSE over both controls in 16/16 folds across four embedding models. |
| Does the direction signal improve held-out retrieval? | Nearest-positive correction with dev-selected step size | No. Retrieval changes have inconsistent sign and usually lower NDCG. |
| Is target mismatch the cause? | Direct linear hard-negative ranking training | No. Corpus features lower training loss, but nonzero held-out corrections lose ranking quality. |
| Can the model learn when correction helps? | Query-only, fixed-reference, and corpus-local utility gates | No. Learned gates fail despite positive oracle-gate headroom. |
| Are prototype cells missing local shape? | Within-cell projected residual means and diagonal variances | No useful increment. The enriched gate is below frozen retrieval in all text-small folds. |
| Can conservative step selection make query-only correction safe? | Per-training-corpus non-degradation rule | No. It selects nonzero steps in 15/16 folds and still harms unseen corpora; strict positive margins mostly collapse to abstention. |
| Is the offline sketch simply too coarse to see the ranking boundary? | Privileged Top-256 score, projected-neighborhood, and candidate-response gate | No transferable evidence. It loses to frozen NDCG in 12/16 folds; utility NMSE is above 1 and correlation is near zero for every model. |

## Interpretation

The decisive separation is between **direction** and **utility**. Corpus-local
features can move a prediction closer to the labeled positive document in the
shared correction space. Retrieval benefit, however, depends on a discontinuous
ordering event among relevant and competing documents. That event does not
transfer across corpora from the tested query, prototype, cell-moment, or even
observed Top-256 boundary features.

This mirrors the distinction made in *Learning Query Encoders Can Be Hard Even
When Vector Retrieval Is Geometrically Easy*: a frozen document index can
support high-quality query vectors without making those vectors learnable from
available supervision. Here the gap remains after conditioning on corpus
geometry. The coherence-based query-performance literature offers stronger
post-retrieval observables, but those require an initial result list and violate
the one-retrieval deployment constraint; the privileged boundary diagnostic
also shows that this information is not sufficient under current cross-corpus
training.

## Scope

The conclusion covers four BEIR-style corpora, four embedding models, shared
low-rank query correction, linear/ridge predictors, ranking-aligned training,
and the tested prototype and boundary summaries. It does not prove an
impossibility theorem for every nonlinear model, corpus family, or source of
supervision. A future result should reopen corpus-aware calibration only if it
introduces a qualitatively new observable and first demonstrates transferable
utility prediction against the frozen and query-only controls.

Increasing network capacity, adding more prototype moments, or trying another
lambda heuristic does not meet that bar. Those changes have no positive
upper-bound evidence and create more ways to fit corpus identity.

## Recommended baseline

- Deploy the unchanged frozen retriever.
- Retain query-only correction as a diagnostic research baseline, not as the
  current reliable policy.
- Preserve corpus-local coordinate prediction as evidence that geometry has
  incremental direction information, while reporting that it does not improve
  held-out retrieval.
- Treat any future corpus-aware method as successful only when it beats both
  frozen retrieval and query-only correction with stable sign under the same
  leave-one-corpus-out protocol.

The research question is therefore closed for the current representation and
supervision families: corpus geometry provides measurable incremental
direction signal, but not a transferable correction policy.
