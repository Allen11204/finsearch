from opensearchpy import OpenSearch

from embeddings import DIMENSION

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

INDEX_NAME = "filings"

mapping = {
    "settings": {
        "index": {"knn": True}   # enable vector search on this index
    },
    "mappings": {
        "properties": {
            "id":      {"type": "keyword"},
            "company": {"type": "keyword"},
            "ticker":  {"type": "keyword"},
            "year":    {"type": "integer"},
            "section": {"type": "keyword"},
            # Decision 1: english analyzer, so "costs increased" matches "cost increase"
            "text":    {"type": "text", "analyzer": "english"},
            "embedding": {
                "type": "knn_vector",
                "dimension": DIMENSION,
                "method": {
                    "name": "hnsw",              # approximate nearest neighbor graph
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                },
            },
        }
    },
}

if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)

client.indices.create(index=INDEX_NAME, body=mapping)
print("Index created:", INDEX_NAME)
