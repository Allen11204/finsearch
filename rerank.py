import sys

from sentence_transformers import CrossEncoder

from search import hybrid_search

# Cross-encoder: reads query and passage together in one forward pass and outputs one relevance score.
# ~90MB, downloaded to ~/.cache/huggingface on first run; max input 512 tokens (query + 200-word chunk fits).
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATES = 50   # how many hybrid results the cross-encoder re-scores
TOP_K = 5         # how many survive; rag.py sends exactly these 5 to the LLM

# Loaded once when the module is imported (~1-2s), then reused for every query.
# rag.py imports this module, so it also gets the already-loaded model.
_model = CrossEncoder(RERANK_MODEL_NAME)


def rerank(query, passages, top_k=TOP_K):
    # passages: list of dicts from search.py, e.g.
    #   [{"id": "UNH_2023_Item7_034", "section": "Item 7", "text": "the estimated potential impact ...",
    #     "score": 0.0315, "company": "UnitedHealth", "ticker": "UNH", "year": 2023}, ...]
    if not passages:
        return []

    # [b] Pair: one (query, passage_text) tuple per candidate, same order as `passages`
    #   [("Did medical costs increase in 2023?", "the estimated potential impact on our medical ..."), ...]
    pairs = [(query, p["text"]) for p in passages]

    # [c] Score: one raw score per pair, aligned by position with `pairs` and `passages`
    #   array([ 6.82, -3.15, 2.40, ...])   shape (50,)
    # Raw scores can be negative and have no fixed range: compare them only within one query.
    scores = _model.predict(pairs)

    # Copy each dict (so the caller's list is not mutated) and attach its score.
    # float(): numpy float32 -> Python float, so the result can be printed / JSON-serialized later.
    # The original "score" (RRF) is kept, so you can compare both orderings.
    scored = [{**p, "rerank_score": float(s)} for p, s in zip(passages, scores)]

    # [d] Sort and cut: highest rerank_score first, keep top_k
    scored.sort(key=lambda p: p["rerank_score"], reverse=True)
    return scored[:top_k]


def search_with_rerank(query, top_k=TOP_K):
    # [a] Get candidates: hybrid (BM25 + vector, fused with RRF) top 50
    #   -> list of 50 dicts, ordered by RRF score
    candidates = hybrid_search(query, size=CANDIDATES)
    # [b]-[d] happen inside rerank(): 50 dicts in -> 5 dicts out, each with "rerank_score"
    return rerank(query, candidates, top_k)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Did medical costs increase in 2023?"
    print("Query:", query)

    # Hybrid alone (no rerank): the order we had before this step
    print("===== Hybrid top 5 (RRF score) =====")
    for r in hybrid_search(query, size=TOP_K):
        print(f'{r["score"]:.4f} | {r["id"]} | {r["text"][:80]}')

    # Hybrid top 50 -> cross-encoder -> top 5
    print("===== Reranked top 5 (rerank_score) =====")
    for r in search_with_rerank(query):
        print(f'{r["rerank_score"]:7.3f} | {r["id"]} | {r["text"][:80]}')
