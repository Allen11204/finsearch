import sys

from config import INDEX_NAME, get_client
from models.embeddings import embed_query

client = get_client()
RRF_K = 60   # the 60 in the RRF formula 1 / (k + rank)


def _to_results(response):
    # Flatten OpenSearch hits into plain records; drop the vector, nobody needs to see 384 floats
    results = []
    for hit in response["hits"]["hits"]:
        r = {k: v for k, v in hit["_source"].items() if k != "embedding"}
        r["score"] = hit["_score"]
        results.append(r)
    return results


def bm25_search(query, size=10):
    # Decision 2: "match" is OR by default, any query word qualifies a chunk, BM25 ranks them
    body = {
        "size": size,
        "query": {"match": {"text": query}},
        "_source": {"excludes": ["embedding"]},
    }
    return _to_results(client.search(index=INDEX_NAME, body=body))


def vector_search(query, size=10):
    body = {
        "size": size,
        "query": {"knn": {"embedding": {"vector": embed_query(query), "k": size}}},
        "_source": {"excludes": ["embedding"]},
    }
    return _to_results(client.search(index=INDEX_NAME, body=body))


def hybrid_search(query, size=10, candidates=50):
    # 1. Take 50 candidates from each side
    bm25_results = bm25_search(query, size=candidates)
    vector_results = vector_search(query, size=candidates)

    # 2. Score each chunk by rank, add up both sides
    scores = {}   # id -> total RRF score
    docs = {}     # id -> chunk record
    for result_list in [bm25_results, vector_results]:
        for rank, r in enumerate(result_list, start=1):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (RRF_K + rank)
            docs[r["id"]] = r

    # 3. Sort by total score, keep the top `size`
    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)

    results = []
    for doc_id in ranked_ids[:size]:
        r = dict(docs[doc_id])
        r["score"] = scores[doc_id]   # replace the original score with the RRF score
        results.append(r)
    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "why did medical costs increase"
    print("Query:", query)

    print("===== BM25 =====")
    for r in bm25_search(query)[:5]:
        print(f'{r["score"]:.4f} | {r["id"]} | {r["text"][:80]}')

    print("===== Vector =====")
    for r in vector_search(query)[:5]:
        print(f'{r["score"]:.4f} | {r["id"]} | {r["text"][:80]}')

    print("===== Hybrid =====")
    for r in hybrid_search(query)[:5]:
        print(f'{r["score"]:.4f} | {r["id"]} | {r["text"][:80]}')
