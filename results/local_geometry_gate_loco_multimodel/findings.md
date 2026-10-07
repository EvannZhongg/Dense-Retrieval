# Benefit gating across embedding models

## Protocol

The fixed-direction benefit-gating experiment was repeated unchanged for
Qwen3, BGE-M3, and E5. Each gate sees the same query-only correction candidate;
only the gate features differ. Candidate lambda, gate ridge penalty, and gate
threshold are selected using training-corpus dev qrels. Held-out qrels are used
only for final metrics and the explicitly diagnostic oracle gate.

## Results

| Model | Corpus gate NDCG wins over baseline | Wins over q-only gate | Wins over reference gate | Mean corpus-gate delta vs baseline | Mean oracle headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | 1/4 | 2/4 | 1/4 | -0.0009 | +0.0070 |
| BGE-M3 | 1/4 | 1/4 | 2/4 | -0.0039 | +0.0166 |
| E5 | 1/4 | 1/4 | 1/4 | +0.0030 | +0.0254 |

E5/ArguAna is a large isolated positive result (HitRate@10 +0.0758 and
NDCG@10 +0.0388), but the same E5 gate decreases NDCG on all three other
held-out corpora. The corpus gate is essentially tied with q-only/reference on
average and has no stable sign. In contrast, the oracle gate has positive mean
headroom for every model, showing that selective correction remains possible
but its utility is not predictable from the current features.

## Conclusion

The failure on text-embedding-3-small is not model-specific. Prototype-center
direction, density, entropy, and posterior variance encode transferable
correction direction but not transferable ranking benefit. Adding predictor
capacity is unsupported. A defensible next representation test must add
information about the local document decision boundary, such as within-cell
projected covariance, while retaining the same q-only and fixed-reference
controls.
