# FinSearch — CLAUDE.md

Shared context for Claude Code. Personal preferences go in `CLAUDE.local.md` (git-ignored).
README.md is the source of truth for architecture, design decisions, and data contracts; this file
only adds what Claude needs on top of it.

## Project

FinSearch is an IS2140 (Information Retrieval) course project: a search and cited Q&A system over
healthcare-sector SEC 10-K filings. Current data: one filing (UnitedHealth, FY2023).

## Code layout

```
Offline
  ingestion/  download.py → extract_text.py → extract_sections.py → chunk.py
  indexing/   create_index.py → index_chunks.py (computes embeddings and writes to OpenSearch)

Online
  search/retrieval.py  bm25_search / vector_search / hybrid_search
  search/rerank.py     rerank / search_with_rerank (cross-encoder over hybrid candidates)
  rag/answer.py        retrieve → prompt → call_llm → validate citations → return
  api/ + frontend/     FastAPI + React
  evaluation/          test queries, LLM-as-judge, nDCG / MRR

Shared
  config.py                paths, OpenSearch client (get_client), INDEX_NAME, model names
  models/embeddings.py     bi-encoder: embed_passages / embed_query
  models/cross_encoder.py  cross-encoder: score_pairs
```

Layout rules:

- `models/` loads a model and encodes; `search/` and `rag/` hold ranking and answer logic.
- Settings used by more than one file go in `config.py`.
- New API or UI code calls the existing functions in `search/` and `rag/`; it does not reimplement retrieval.

## Running

Run every script as a module from the project root:

```bash
python -m search.retrieval "medical cost trend increase"
python -m search.rerank "Did medical costs increase in 2023?"
python -m rag.answer "Why did medical costs increase in 2023?"
```

Plain `python search/rerank.py` fails with `No module named 'config'`.
OpenSearch must be up (`docker compose up -d`); `rag.answer` also needs `ollama serve`.

## Rules for changes

- Before editing, read the files involved to confirm current function names and signatures.
- Make small changes; don't rewrite files that already work.
- Don't change anything in README's **Design decisions** or **Data contracts** without asking first:
  evaluation labels, cached results, and the API depend on them.
- After changing ingestion, indexing, or search code, rerun the affected scripts and compare output
  with `results/mid_demo_runs.txt`.
- All code, comments, output strings, docs, and commit messages are in English.

## Notes not in README

- Reranker: compare `cross-encoder/ms-marco-MiniLM-L-6-v2` with `BAAI/bge-reranker-base` during evaluation.
- LLM at deployment is undecided (local on EC2 vs an API such as AWS Bedrock). Only `rag.answer.call_llm` changes.
- 5 passages go to the LLM: 3 risks missing the answer, 10 adds noise and cost.
- Prompt: rules in the system prompt; numbered passages first, question last in the user message.
- If the LLM ignores the citation rule or invents answers, report examples before changing the model or prompt.
