import re
import sys
import time

import ollama

from config import LLM_MODEL
from search.rerank import search_with_rerank

# LLM_MODEL (config.py) is served by Ollama at http://localhost:11434
TOP_K = 5   # passages sent to the LLM; citations [1]..[5] are valid

# Rules live in the system prompt: models follow system rules more reliably than rules mixed into the user turn.
SYSTEM_PROMPT = (
    "You answer questions about company annual reports (10-K filings).\n"
    "Answer using only the passages provided by the user.\n"
    "Cite the passage number for each claim, like [1].\n"
    "If the passages don't contain the answer, say \"I couldn't find this in the filings.\""
)

# Matches "[3]" and captures "3". Does NOT match "[1, 2]" or "[1-3]": those would show up as
# missing citations, which is a format error we want to see during evaluation, not hide.
CITATION_RE = re.compile(r"\[(\d+)\]")

# Safety net: if thinking mode leaks through anyway, drop "<think>...</think>" from the answer.
# re.DOTALL lets "." match newlines, since the think block spans many lines.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_user_message(question, passages):
    # [2] Build prompt: numbered passages first, question last.
    # The number n here is what the LLM writes as [n]; [4] maps it back to passages[n-1].
    # Output shape:
    #   [1] (UNH_2023_Item7_034) the estimated potential impact on our medical costs ...
    #
    #   [2] (UNH_2023_Item7_030) the health care professional and type of service ...
    #   ...
    #   Question: Why did medical costs increase in 2023?
    blocks = [f'[{n}] ({p["id"]}) {p["text"]}' for n, p in enumerate(passages, start=1)]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def call_llm(user_message):
    # [3] Generate. The only function that knows the LLM provider:
    # switching to an API (e.g. AWS Bedrock) at deploy time means rewriting only this function.
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # temperature=0: always pick the most likely next token -> same question gives the same answer
        # num_predict=500: hard cap on output tokens, so a rambling answer cannot run for minutes
        options={"temperature": 0, "num_predict": 500},
        # Qwen3 "thinking mode" off: no <think> reasoning text before the answer (and much faster)
        think=False,
    )
    # response["message"]["content"] -> "Medical costs increased due to ... [1][2]."
    return THINK_RE.sub("", response["message"]["content"]).strip()


def validate_citations(answer, num_passages):
    # [4] Validate: every [n] the LLM wrote must point to a passage we actually sent (1..num_passages).
    # "Costs rose [1][3]. Also [7]." -> numbers in order of appearance: [1, 3, 7]
    numbers = [int(n) for n in CITATION_RE.findall(answer)]
    # dict.fromkeys removes duplicates and keeps first-seen order: [1, 3, 1] -> [1, 3]
    unique = list(dict.fromkeys(numbers))
    valid = [n for n in unique if 1 <= n <= num_passages]        # -> [1, 3]
    invalid = [n for n in unique if not 1 <= n <= num_passages]  # -> [7]  (no passage 7 exists)
    return valid, invalid


def answer(question):
    # [1] Retrieve: hybrid top 50 -> cross-encoder -> top 5 dicts (id, section, text, rerank_score, ...)
    passages = search_with_rerank(question, top_k=TOP_K)

    # [2] Build prompt
    user_message = build_user_message(question, passages)

    # [3] Generate, timed: local 8B models are slow, and this number goes on the slides.
    # The first call after starting Ollama also includes loading the model into memory.
    start = time.perf_counter()
    answer_text = call_llm(user_message)
    generation_seconds = time.perf_counter() - start

    # [4] Validate
    valid, invalid = validate_citations(answer_text, len(passages))

    # [5] Return. citations: the passage dicts the answer actually cited, with "n" = the number used in
    # the answer, so a UI can render "[2]" next to the right source.
    #   {"question": "...", "answer": "... [1] ... [3].",
    #    "citations": [{"n": 1, "id": "UNH_2023_Item7_034", ...}, {"n": 3, ...}],
    #    "invalid_citations": [], "generation_seconds": 14.2}
    return {
        "question": question,
        "answer": answer_text,
        "citations": [{"n": n, **passages[n - 1]} for n in valid],
        "invalid_citations": invalid,
        "generation_seconds": generation_seconds,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Why did medical costs increase in 2023?"
    result = answer(question)

    print("Question:", result["question"])
    print("===== Answer =====")
    print(result["answer"])
    print("===== Cited passages =====")
    for c in result["citations"]:
        print(f'[{c["n"]}] {c["id"]} | {c["text"][:80]}')
    if not result["citations"]:
        print("(none)")
    print("Invalid citations:", result["invalid_citations"] or "none")
    print(f'Generation time: {result["generation_seconds"]:.1f}s')
