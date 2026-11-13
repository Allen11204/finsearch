import sys

from models.cross_encoder import score_pairs
from search.retrieval import hybrid_search

CANDIDATES = 50   # how many hybrid results the cross-encoder re-scores
TOP_K = 5         # how many survive; rag/answer.py sends exactly these 5 to the LLM


def rerank(query, passages, top_k=TOP_K):
    # passages: list of dicts from search/retrieval.py, e.g.
    #   [{"id": "UNH_2023_Item7_034", "section": "Item 7", "text": "the estimated potential impact ...",
    #     "score": 0.0315, "company": "UnitedHealth", "ticker": "UNH", "year": 2023}, ...]
    if not passages:
        return []

    # [b] Pair: one (query, passage_text) tuple per candidate, same order as `passages`
    #   [("Did medical costs increase in 2023?", "the estimated potential impact on our medical ..."), ...]
    pairs = [(query, p["text"]) for p in passages]

    # [c] Score: one raw score per pair, aligned by position with `pairs` and `passages`
    #   array([ 6.82, -3.15, 2.40, ...])   shape (50,)  (see models/cross_encoder.py)
    scores = score_pairs(pairs)

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
