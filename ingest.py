"""Chunk, embed and index the coffee guides (.md / .txt) into Elasticsearch.

Markdown files are split by headings, so each chunk covers one topic (for example
"French press" or "Check the roast date first"). Long sections are split further into
overlapping word windows.

Usage:
    python ingest.py docs/              # add or update documents
    python ingest.py docs/ --recreate   # drop the index and start fresh
"""
import argparse
import hashlib
import re
from pathlib import Path

from elasticsearch import helpers

from common import CHUNK_OVERLAP, CHUNK_WORDS, INDEX, embed, embedding_dims, get_es

SUPPORTED = {".txt", ".md"}
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def window_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows (used for long sections and plain text)."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (document title, [(section heading path, section body), ...]) for markdown."""
    title = ""
    sections: list[tuple[str, str]] = []
    path: list[str] = []          # heading stack below the H1, e.g. ["Pour-over"]
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(path), content))
        body.clear()

    for line in text.splitlines():
        m = HEADING.match(line)
        if not m:
            body.append(line)
            continue
        flush()
        level, heading = len(m.group(1)), m.group(2)
        if level == 1:
            title, path = heading, []
        else:
            path = path[: level - 2] + [heading]
    flush()
    return title, sections


def chunk_document(text: str, filename: str) -> list[dict]:
    """Return chunks as dicts with title, section and text.

    The title and section are prepended to the chunk text so both keyword and vector
    search can match on them ("french press" finds the French press section).
    """
    title, sections = split_sections(text)
    title = title or Path(filename).stem.replace("-", " ")
    chunks = []
    for section, body in sections or [("", text)]:
        for piece in window_words(body):
            header = f"{title} - {section}" if section else title
            chunks.append({"title": title, "section": section, "text": f"{header}\n{piece}"})
    return chunks


def ensure_index(es, dims: int, recreate: bool) -> None:
    if es.indices.exists(index=INDEX):
        if not recreate:
            return
        es.indices.delete(index=INDEX)
    es.indices.create(
        index=INDEX,
        mappings={
            "properties": {
                "text": {"type": "text", "analyzer": "english"},     # BM25 keyword search
                "title": {"type": "text", "analyzer": "english"},
                "section": {"type": "text", "analyzer": "english",
                            "fields": {"raw": {"type": "keyword"}}},
                "source": {"type": "keyword"},                        # file path, for citations
                "chunk_id": {"type": "integer"},
                "embedding": {                                        # vector (kNN) search
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )
    print(f"Created index '{INDEX}' ({dims}-dim vectors)")


def iter_files(folder: str):
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder of .txt/.md files to index")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the index")
    args = parser.parse_args()

    es = get_es()
    ensure_index(es, embedding_dims(), args.recreate)

    total = 0
    for path in iter_files(args.folder):
        chunks = chunk_document(path.read_text(encoding="utf-8", errors="ignore"), path.name)
        if not chunks:
            continue
        source = str(path)

        # Remove old chunks for this file so re-ingesting an edited file leaves no stale chunks.
        es.delete_by_query(index=INDEX, query={"term": {"source": source}}, refresh=True)

        vectors = embed([c["text"] for c in chunks])
        actions = [
            {
                "_index": INDEX,
                "_id": hashlib.sha1(f"{source}:{i}".encode()).hexdigest(),
                "_source": {**chunk, "source": source, "chunk_id": i, "embedding": vec},
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        ok, errors = helpers.bulk(es, actions, raise_on_error=False)
        if errors:
            print(f"  {source}: {len(errors)} errors, first: {errors[0]}")
        print(f"Indexed {ok} chunks from {source}")
        total += ok

    es.indices.refresh(index=INDEX)
    print(f"Done. {total} chunks indexed into '{INDEX}'.")


if __name__ == "__main__":
    main()
