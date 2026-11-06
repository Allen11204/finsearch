# FinSearch — CLAUDE.md

## Project

FinSearch is an IS2140 (Information Retrieval) course project: a search and cited Q&A system over healthcare-sector SEC 10-K filings. Current data: one filing (UnitedHealth, FY2023).

Current goal: get reranking and RAG running end to end before the mid presentation (due Nov 11, 2026). "Running" means it executes and produces output. Do NOT tune quality yet; quality is measured later in the evaluation stage.

## How to work with Allen (the developer)

- Allen is learning. Reply in Chinese, and give the English term for every concept (e.g. 重排序 rerank, 交叉编码器 cross-encoder).
- Before writing code for a step: state the requirement, then the approach, then the code.
- Every code block must carry inline comments explaining what each piece does, why, and how it connects to the pieces before and after. Show example values (real data shapes) in comments.
- No analogies. Explain with the actual code and a concrete example.
- Small steps. Do not rewrite files that already work.

## Pipeline map

```
Offline (done)
  download.py → extract_text.py → extract_sections.py → chunk.py
  → create_index.py → index_chunks.py (computes embeddings and writes to OpenSearch)

Online
  8.  Retrieve   search.py      bm25_search / vector_search / hybrid_search   DONE
  9.  Rerank     rerank.py      cross-encoder over hybrid candidates           ← TASK 1
  10. Generate   rag.py         retrieve → prompt → LLM → validate → return    ← TASK 2
  11. Present    FastAPI + React                                              later
  12. Evaluate   test queries, LLM-as-judge, nDCG / MRR                       later
```

Before changing anything, read search.py, embeddings.py, create_index.py, and index_chunks.py to confirm current function names and signatures. Expected (verify):

- embeddings.py: `embed_passages(texts)`, `embed_query(query)` using BAAI/bge-small-en-v1.5
- search.py: `bm25_search(query, size)`, `vector_search(query, size)`, `hybrid_search(query, size=10, candidates=50)`; each returns a list of dicts with id, score, section, text

## Locked decisions (do not change without asking Allen)

| Area | Decision | Reason |
|---|---|---|
| Text analysis | OpenSearch english analyzer on text | matches word forms (costs/cost) |
| Matching | BM25 match with OR | NL questions contain words passages don't use |
| Embedding model | BAAI/bge-small-en-v1.5, 384 dims, normalized; query prefix "Represent this sentence for searching relevant passages: " | 512-token limit fits 200-word chunks; fast on CPU |
| Vector index | knn_vector, HNSW, cosinesimil, lucene engine | simple, supports filtering |
| Fusion | RRF, k = 60, 50 candidates per side | scores are on different scales |
| Rerank model | cross-encoder/ms-marco-MiniLM-L-6-v2 | fast on CPU (small EC2), no per-call cost; compare with BAAI/bge-reranker-base in evaluation |
| Rerank candidates | top 50 from hybrid → keep top 5 | matches retrieval size; ~1s on CPU |
| Passages to LLM | 5 | 3 risks missing the answer; 10 adds noise and cost |
| LLM | Local via Ollama for development and the mid presentation (free, no API key). Default model qwen3:8b; use qwen3:4b if the Mac has 8GB RAM or answers are too slow | no cost during development |
| LLM at deployment | Undecided. A CPU EC2 running an 8B model is slow and needs a large instance; decide at deploy time between local-on-EC2 and an API (e.g. AWS Bedrock) | only call_llm changes |
| LLM call | isolated in one function `call_llm(user_message)` | switching provider changes only this function |
| Prompt | rules in the system prompt; numbered passages first, question last in the user message | system rules are followed more reliably |
| Generation params | temperature=0, max 500 output tokens, Qwen3 thinking mode off | reproducible answers; no `<think>` text in answers |
| Citations | LLM cites [n]; code validates n is in 1..5 | catches citations to non-existent passages; small local models break format more often |

## TASK 1 — Rerank (rerank.py)

Framework (label code sections with these):

- [a] Get candidates: `hybrid_search(query, size=50)`
- [b] Pair: build (query, passage_text) for each of the 50
- [c] Score: `CrossEncoder.predict(pairs)` → 50 scores (raw scores, can be negative; compare only within one query)
- [d] Sort and cut: sort by score desc, keep top_k (5)

Steps:

1. Confirm sentence-transformers is installed (it is, for embeddings).
2. Create rerank.py:
   - load `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` once at module level
   - `rerank(query, passages, top_k)` → passages with added rerank_score, sorted, top_k
   - `search_with_rerank(query, top_k=5)` → [a] then rerank
   - `__main__`: print hybrid top 5 and reranked top 5 side by side for sys.argv[1]
3. Run:
   ```bash
   python rerank.py "Did medical costs increase in 2023?"
   ```

Done when: both lists print, and reranked list shows rerank_score values. First run downloads ~90MB.

## TASK 2 — RAG (rag.py)

Framework (label code sections with these):

- [1] Retrieve: `search_with_rerank(question, top_k=5)`
- [2] Build prompt: system prompt (rules) + user message (`[1] (id) text` … then `Question: ...`)
- [3] Generate: `call_llm(user_message)` — the only place that knows the provider
- [4] Validate: regex `\[(\d+)\]` → split into valid (1..5) and invalid
- [5] Return: `{question, answer, citations: [passage dicts], invalid_citations}`

System prompt:

```
You answer questions about company annual reports (10-K filings).
Answer using only the passages provided by the user.
Cite the passage number for each claim, like [1].
If the passages don't contain the answer, say "I couldn't find this in the filings."
```

Steps:

1. Ollama setup (Allen does this in the terminal; explain each command):
   ```bash
   brew install ollama        # or download from ollama.com
   ollama serve               # keep running in its own terminal (or run the Ollama app)
   ollama pull qwen3:8b       # ~5GB download; use qwen3:4b on an 8GB Mac
   ollama run qwen3:8b "say hi"   # sanity check, then exit
   ```
2. `pip install ollama`; add it to requirements.txt.
3. Create/update rag.py per the framework above. call_llm uses the ollama Python client:
   ```python
   ollama.chat(
       model=LLM_MODEL,
       messages=[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_message}],
       options={"temperature": 0, "num_predict": 500},
       think=False,
   )
   ```
   and returns `response["message"]["content"]`. If the installed Ollama version rejects `think=False`, append `/no_think` to the end of the user message instead, and strip any `<think>...</think>` block from the answer as a fallback.
4. `__main__`: print the answer, the cited passages (id | first 80 chars), and invalid citations. Also print how many seconds generation took (local models are slow; Allen needs this number).
5. Run three checks:
   ```bash
   python rag.py "Why did medical costs increase in 2023?"        # answer in passages
   python rag.py "What risks does UnitedHealth face from regulation?"  # Item 1A
   python rag.py "What was Apple's revenue in 2023?"               # NOT in data → should say it couldn't find it
   ```

Done when: all three run without errors; the first two cite valid passages; the third declines instead of inventing an answer. Save the terminal output; Allen needs it for slides.

If the model ignores the citation rule or invents answers often, report it to Allen with examples before changing models or prompts.

## Out of scope for now

- Tuning prompts, models, chunk size, or RRF k
- FastAPI, React, Redis, AWS deployment
- Scaling to more companies
- Moving embedding out of index_chunks.py into a separate embed.py (planned later, before scaling data)

## After both tasks

Report to Allen (in Chinese with English terms):

- Output of `python rerank.py "Did medical costs increase in 2023?"`
- Output of the three rag.py runs
- Any errors hit and how they were fixed
