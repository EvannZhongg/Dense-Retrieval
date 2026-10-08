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
| Is the negative result only caused by nearest-positive regression targets? | LOCO direct prediction of privileged hard-negative ranking coordinates in a rank-32 boundary basis | No. Query-only coordinates remain unreliable, and a document-only spectral interaction is worse than query-only in all text-small and Qwen3 folds. |
| Are privileged ranking coordinates non-identifiable because many orthogonal optima give the same retrieval result? | Repeated per-query oracle optimization under initialization, optimizer-seed, negative-sample/count, and regularization perturbations | Not generally. Fixed-objective reruns are strongly aligned; negative construction changes both the direction and the resulting ranking. |
| Does scoring several actions by their actual retrieval utility avoid brittle coordinate targets? | LOCO candidate-action NDCG supervision with exact abstention and matched query/reference/corpus features | No. The candidate oracle has mean `+0.0302` NDCG headroom, but q-only loses in 4/4 folds and corpus geometry has 1 win, 1 tie, and 2 losses. |

## Interpretation

The decisive separation is between **direction** and **utility**. Corpus-local
features can move a prediction closer to the labeled positive document in the
shared correction space. Retrieval benefit, however, depends on a discontinuous
ordering event among relevant and competing documents. That event does not
transfer across corpora from the tested query, prototype, cell-moment, or even
observed Top-256 boundary features.

The action-space audit closes a separate loophole. With held-out qrels available
to optimize one coordinate vector per query, a rank-32 subspace retains
substantial ranking capacity (about 76-82% of the full-ambient oracle gain in
the tested text-small folds, depending on the basis). A rank-64 subspace retains
about 85%. Therefore the earlier predictor failures are not evidence that rank
32 is intrinsically too small. Directly changing the training target to those
privileged ranking coordinates also does not recover transfer: the q-only model
is below frozen retrieval on mean NDCG, while the simple document-spectrum
interaction is substantially worse. This is a target/predictor generalization
failure, not a missing PCA variance explanation.

The random rank-32 oracle result must be interpreted only as an existence or
capacity statement. Its 82.3% gain retention says that a privileged optimizer
can often find some useful direction when given 32 degrees of freedom. It does
not show that the subspace contains a stable, semantically consistent, or
predictable correction direction, and oracle retention must not be used as a
proxy for representation learnability.

The ranking-optimum stability audit tests one possible explanation for that
gap. For text-small, repeated initialization yields mean coordinate cosine
`0.9798`; changing only the optimizer seed is exactly stable, and small
regularization changes yield `0.9973`. Negative-sample and negative-count
perturbations reduce cosine to `0.9043` and `0.9339`, but also reduce mean
Top-10 overlap to `0.6718` and `0.6669` and change the common-reference ranking
loss. No directionally distinct pair (cosine below `0.2`) simultaneously keeps
Top-10 overlap above `0.8` and NDCG/MRR differences below `0.01`. Therefore the
data do not support widespread orthogonal, retrieval-equivalent optima inside
the fixed objective. They instead expose sensitivity of the target itself to
hard-negative construction. Coordinate regression may still be a brittle
target across corpora, but optimizer non-identifiability is not the primary
explanation established by this experiment.

Candidate-action utility supervision removes the remaining assumption that a
policy must reproduce one privileged coordinate vector. A training-only action
codebook offers zero correction plus 24 direction/magnitude choices, and the
scorer is trained on each action's actual per-query NDCG gain. The candidate
oracle retains mean held-out headroom of `+0.0302`, so useful discrete actions
do exist. Nevertheless, query-only scoring lowers NDCG in all four text-small
folds, while corpus-local scoring has only one win, one tie, and two losses and
lowers mean NDCG by `0.00216`. The bottleneck therefore persists when the
supervision is retrieval utility itself and the policy can abstain exactly.

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

The direct-coordinate follow-up covers four corpora with text-small and Qwen3.
It uses a rank-32 ranking-boundary basis, document-only shared spectral
statistics, and strict leave-one-corpus-out fitting. It is a falsification of
that new representation family, not an impossibility theorem for arbitrary
query-dependent manifolds.

The candidate-action follow-up is intentionally restricted to the fixed
text-small embedding space required by the current stage. It tests a linear
utility scorer over a training-only discrete action codebook and prototype
geometry. It rules out that matched design, not arbitrary action sets or
nonlinear policies.

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
