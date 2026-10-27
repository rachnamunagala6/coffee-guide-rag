"""Coffee guide: ask how to pick beans at the store and how to brew them.

Answers come from the guides in docs/, found with hybrid search in Elasticsearch.

Usage:
    python ask.py "what should I look for on a bag of beans?"
    python ask.py            # interactive mode
"""
import sys

import anthropic

from common import LLM_MODEL
from search import hybrid_search

SYSTEM_PROMPT = """You are a friendly, practical coffee guide. You help people choose coffee beans \
at the store and turn them into great coffee at home: reading labels, picking a roast and origin \
they will enjoy, storing and grinding beans, and brewing with any method.

How to answer:
- Base your answer on the numbered guide passages provided. Cite the ones you use with bracketed \
numbers like [1] or [2][3].
- Lead with the direct answer, then the practical details: amounts, grind size, temperatures, times \
and steps. Use a short numbered list for brewing steps.
- Give measurements in both grams and cups/tablespoons or ounces where the passages allow, and \
temperatures in both °F and °C.
- If the person's question is vague (for example "which beans should I buy?"), give a good default \
recommendation and mention the one or two things that would change it, such as how they brew or \
whether they like it fruity or chocolatey.
- If the passages do not cover the question, say so plainly and do not invent specifics. Questions \
unrelated to coffee: briefly say you can only help with choosing and brewing coffee.
- Keep it concise and conversational."""

EXAMPLES = [
    "What should I look for on a bag of coffee at the grocery store?",
    "I like smooth, chocolatey coffee. Which beans should I buy?",
    "How do I make coffee with a French press?",
    "My pour-over tastes sour. What am I doing wrong?",
    "Should I keep my beans in the freezer?",
]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}]\n{c['text']}" for i, c in enumerate(chunks, start=1))


def answer(question: str, client: anthropic.Anthropic,
           history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Answer a question. `history` is earlier turns: [{"role": "user"|"assistant", "content": str}].

    For follow-ups ("what about for espresso?"), the previous question is added to the search
    query so retrieval knows what "it" refers to.
    """
    history = history or []
    previous_questions = [t["content"] for t in history if t["role"] == "user"]
    search_query = f"{previous_questions[-1]} {question}" if previous_questions else question

    chunks = hybrid_search(search_query)
    if not chunks:
        return "The coffee guides aren't indexed yet. Run: python ingest.py docs/ --recreate", []
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            *history[-6:],   # last 3 exchanges, enough for follow-ups without growing cost
            {
                "role": "user",
                "content": f"<guides>\n{build_context(chunks)}\n</guides>\n\nQuestion: {question}",
            },
        ],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return text, chunks


def run(question: str, client: anthropic.Anthropic) -> None:
    text, chunks = answer(question, client)
    print(f"\n{text}\n")
    if chunks:
        print("Sources:")
        for i, c in enumerate(chunks, start=1):
            where = f"{c.get('title', '')} > {c.get('section', '')}".strip(" >")
            print(f"  [{i}] {where}")
        print()


if __name__ == "__main__":
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment / .env
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]), client)
    else:
        print("Coffee guide - ask about choosing beans or brewing. Try:")
        for ex in EXAMPLES:
            print(f"  - {ex}")
        while True:
            q = input("\nQuestion (blank to quit): ").strip()
            if not q:
                break
            run(q, client)
