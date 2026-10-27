"""Web front end for the coffee guide.

Usage:
    python app.py           # then open http://localhost:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import uvicorn
from elastic_transport import ConnectionError as ESConnectionError
from elasticsearch import NotFoundError
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ask import answer
from common import ES_URL, get_embedder

STATIC = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(_app):
    # Load the embedding model at startup so the first question isn't slow.
    get_embedder()
    yield


app = FastAPI(title="Coffee guide", lifespan=lifespan)
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env (loaded in common.py)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = []


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/ask")
def ask(q: Question):
    try:
        text, chunks = answer(q.question.strip(), client, [t.model_dump() for t in q.history])
    except ESConnectionError:
        return error(503, f"Can't reach Elasticsearch at {ES_URL}. Open Docker Desktop, "
                          "then run: docker compose up -d")
    except NotFoundError:
        return error(503, "The coffee guides aren't indexed yet. Run: python ingest.py docs/ --recreate")
    except anthropic.AuthenticationError:
        return error(502, "Claude rejected the API key. Check ANTHROPIC_API_KEY in .env, "
                          "then restart the app.")
    except TypeError as e:
        if "authentication" in str(e):
            return error(502, "No API key found. Add ANTHROPIC_API_KEY to .env, then restart the app.")
        raise
    except anthropic.BadRequestError as e:
        if "credit" in str(e).lower():
            return error(502, "Your Anthropic account is out of credit. Add credit under "
                              "Billing at console.anthropic.com.")
        raise
    except anthropic.APIConnectionError:
        return error(502, "Couldn't reach Claude. Check your internet connection and try again.")

    sources = [
        {"n": i, "title": c.get("title", ""), "section": c.get("section", "")}
        for i, c in enumerate(chunks, start=1)
    ]
    return {"answer": text, "sources": sources}


if __name__ == "__main__":
    print("Coffee guide running at http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
