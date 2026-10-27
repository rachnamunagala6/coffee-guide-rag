"""Hybrid retrieval: BM25 + kNN vector search, merged with Reciprocal Rank Fusion.

RRF is done client-side so it works on any Elasticsearch license tier.

Usage:
    python search.py "what grind size for french press?"
"""
import sys

from common import INDEX, TOP_K, embed, get_es


def bm25_search(es, query: str, k: int) -> list[dict]:
    # Section headings get extra weight: a question about "french press" should rank the
    # French press section above passing mentions elsewhere.
    resp = es.search(
        index=INDEX,
        query={"multi_match": {"query": query, "fields": ["text", "section^2", "title"]}},
        size=k,
        source_excludes=["embedding"],
    )
    return resp["hits"]["hits"]


def knn_search(es, query_vector: list[float], k: int, num_candidates: int = 100) -> list[dict]:
    resp = es.search(
        index=INDEX,
        knn={"field": "embedding", "query_vector": query_vector, "k": k,
             "num_candidates": max(num_candidates, k)},
        size=k,
        source_excludes=["embedding"],
    )
    return resp["hits"]["hits"]


def rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit["_id"]] = scores.get(hit["_id"], 0.0) + 1.0 / (k + rank)
            docs[hit["_id"]] = hit["_source"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [{"id": doc_id, "score": scores[doc_id], **docs[doc_id]} for doc_id in ranked]


def hybrid_search(query: str, top_k: int = TOP_K, candidates: int = 20, es=None) -> list[dict]:
    es = es or get_es()
    keyword_hits = bm25_search(es, query, candidates)
    vector_hits = knn_search(es, embed([query])[0], candidates)
    return rrf([keyword_hits, vector_hits])[:top_k]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Query: ")
    for i, hit in enumerate(hybrid_search(q), start=1):
        preview = hit["text"][:200].replace("\n", " ")
        where = f"{hit.get('title', '')} > {hit.get('section', '')}".strip(" >")
        print(f"[{i}] {hit['score']:.4f}  {where}\n    {preview}...\n")
