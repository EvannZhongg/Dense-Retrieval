# Candidate-action utility LOCO

## Hypothesis

Direct coordinate regression may be unnecessarily brittle because retrieval
can admit more than one useful correction. This experiment instead learns a
small discrete action set from privileged ranking coordinates on the training
corpora and supervises a scorer with each action's actual per-query NDCG@10
gain. Zero correction is always an available action. The matched scorers use
query-only, fixed-reference geometry, or held-corpus document geometry.

All four folds use `text-embedding-3-small (Aiberm)`. The ranking-boundary
basis, action directions, Ridge model, regularization, and abstention threshold
are fitted or selected using the other three corpora only. Held-out documents
are used to construct the document-only corpus geometry; held-out qrels are
used only for final policy metrics and the explicitly privileged candidate
oracle. A deployed policy chooses one action before its single final retrieval.

## Held-out NDCG@10

| held-out corpus | frozen | q-only | fixed reference | corpus geometry | candidate oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiQA | 0.4484 | 0.4471 | 0.4460 | 0.4484 | 0.4983 |
| ArguAna | 0.3908 | 0.3857 | 0.3909 | 0.3908 | 0.4321 |
| SciFact | 0.8198 | 0.8178 | 0.8037 | 0.8096 | 0.8228 |
| NFCorpus | 0.3245 | 0.3240 | 0.3230 | 0.3262 | 0.3512 |
| mean | 0.4959 | 0.4936 | 0.4909 | 0.4937 | 0.5261 |

Relative to frozen retrieval, q-only is worse in all four folds with mean
NDCG delta `-0.00226`. Corpus geometry has one win, one exact tie, and two
losses, with mean delta `-0.00216`. It slightly exceeds q-only on the macro
mean but does not provide a stable or deployable increment. The fixed-reference
capacity control is worse still (`-0.00499`).

Corpus geometry exceeds q-only in FiQA (`+0.00125`), ArguAna (`+0.00515`), and
NFCorpus (`+0.00218`), but loses sharply in SciFact (`-0.00820`). Its mean
increment over q-only is only `+0.00010`, so the corpus signal does not produce
a consistent utility gain even relative to the weaker correction baseline.

The learned policies do not fail by always abstaining. Mean correction rates
are 49.9% for q-only, 54.6% for fixed-reference, and 36.6% for corpus geometry;
their mean correction magnitudes are 0.099, 0.101, and 0.072. Dev-selected
nonzero actions therefore remain common, but their utility does not transfer.

## Interpretation

The candidate set has meaningful held-out capacity: its privileged mean oracle
gain is `+0.0302` NDCG, with substantial headroom on FiQA, ArguAna, and
NFCorpus. The negative policy result therefore cannot be attributed simply to
the discrete action set containing no useful correction. Directly supervising
retrieval utility and permitting abstention also do not solve the transfer
problem with the currently observable prototype geometry.

This closes the specific loophole that a unique privileged coordinate target
was the main cause of failure. It does not prove that every possible action
representation or nonlinear utility model must fail. A follow-up would need a
qualitatively new pre-retrieval observable, not just more candidate directions,
features, or scorer capacity.
