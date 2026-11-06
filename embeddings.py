from sentence_transformers import SentenceTransformer

# 384-dim English model; max input is 512 tokens, so a 200-word chunk fits without truncation
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384

# bge models expect this prefix on queries (not on passages) for retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    # Load once and reuse: loading takes seconds, encoding a query takes milliseconds
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts):
    # Normalized vectors, so cosine similarity equals dot product
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()


def embed_query(query):
    return get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
