from sentence_transformers import CrossEncoder

from config import RERANK_MODEL

# Cross-encoder: reads query and passage together in one forward pass and outputs one relevance score.
# ~90MB, downloaded to ~/.cache/huggingface on first run; max input 512 tokens (query + 200-word chunk fits).

_model = None


def get_model():
    # Load once and reuse (~1-2s to load), same pattern as models/embeddings.py
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def score_pairs(pairs):
    # pairs: [("Did medical costs increase in 2023?", "the estimated potential impact on our medical ..."), ...]
    # -> one raw score per pair, same order:  array([ 6.82, -3.15, 2.40, ...])
    # Raw scores can be negative and have no fixed range: compare them only within one query.
    return get_model().predict(pairs)
