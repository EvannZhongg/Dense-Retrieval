# Direct ranking-coordinate LOCO: Qwen3

The same protocol was repeated with Qwen3-Embedding. Mean NDCG@10 was `0.5061`
for frozen retrieval, `0.5038` for q-only correction, and `0.4302` for the
corpus-spectral predictor. Corpus spectral correction underperformed frozen
retrieval on all four held-out corpora; q-only correction had no stable sign.

This replication supports the text-small finding that the tested document-only
spectral interaction does not provide a transferable ranking-coordinate signal.
