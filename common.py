"""Shared config, Elasticsearch client and embedding helpers."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ES_API_KEY")
INDEX = os.getenv("ES_INDEX", "coffee-guide")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))


def get_es() -> Elasticsearch:
    if ES_API_KEY:
        return Elasticsearch(ES_URL, api_key=ES_API_KEY)
    return Elasticsearch(ES_URL)


@lru_cache(maxsize=1)
def get_embedder():
    # Imported lazily because it pulls in torch, which is slow to load.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Return L2-normalised embeddings (pairs with cosine similarity in the index)."""
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embedding_dims() -> int:
    return get_embedder().get_sentence_embedding_dimension()
