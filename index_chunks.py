import json

from opensearchpy import OpenSearch, helpers

from embeddings import embed_passages

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

INDEX_NAME = "filings"

with open("data/chunks/UNH_2023_chunks.json", encoding="utf-8") as f:
    records = json.load(f)

# Embed all chunks in one batch call, much faster than one call per chunk
vectors = embed_passages([record["text"] for record in records])

# Use the record id as the document _id, so re-running overwrites instead of duplicating
actions = [
    {"_index": INDEX_NAME, "_id": record["id"], "_source": {**record, "embedding": vector}}
    for record, vector in zip(records, vectors)
]

success, errors = helpers.bulk(client, actions)
print("Indexed:", success, "| errors:", errors)

# New documents become searchable after a refresh (OpenSearch does this every 1s by default)
client.indices.refresh(index=INDEX_NAME)
print("Documents in index:", client.count(index=INDEX_NAME)["count"])
