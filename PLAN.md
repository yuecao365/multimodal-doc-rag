# Plan: phases 5–9

Read the "Main result" and "Fixed setup" sections, then only the section for the
phase you are working on. Phases run in order; finish one (tests pass, ladder
updated, decisions recorded, committed) before starting the next.

## Main result: an improvement ladder

The project's headline is: a simple, standard RAG baseline scores X; a sequence of
design changes, each measured, raises it to Y on held-out data.

- **Step 0, baseline:** the simplest standard RAG. Text route alone (parsed text,
  chunks, dense embeddings, LightRAG naive mode), top 5 pages given to the
  generator as text. It must be a fair, standard baseline: same generator, judge
  and page budget as every later step, no handicaps.
- **Each later step adds one change on top of the current best system** and is
  measured on the development set (hr).
- **No assumptions about what will help.** A change stays in the system only if it
  improves the primary metric on hr by more than noise: a paired randomization test
  over queries, p < 0.05, reported with the difference and its 95% confidence
  interval. Otherwise it is recorded as a negative result and later steps do not
  build on it.
- **Primary metrics:** NDCG@10 for retrieval changes; answer score (judge) for
  changes to generation or to the agent. Secondary: index storage per page, cost
  and latency per query.
- **Efficiency changes** (per-page routing, compression, Matryoshka) are not judged
  on score. They stay if quality does not drop beyond noise while storage or cost
  falls; report the saving.
- **Ladder table:** keep `results/ladder.md` up to date after every run: step,
  change, dev NDCG@10, dev answer score (when run), storage per page, cost per
  query, p-value, kept or not, link to the run directory.
- **The ladder order matters for attribution,** so phase 9 also removes each kept
  change from the final system one at a time (leave-one-out ablation).
- **Answer runs are paid,** so the judge scores answers only where they decide
  something: on hr the baseline, the final system, the agent vs the best non-agent
  system, and the robustness checks (phase 8); on finance_en the baseline and the
  final system. Retrieval changes, including their leave-one-out ablations, are
  measured with NDCG@10 and Recall@10 only.
- Every technique is one switch in `PipelineSpec` (or one experiment script), off
  by default. Keep only what supports a conclusion in the final README; do not add
  variants, backends or models just to have them.

## Fixed setup

- **Data roles.** ViDoRe has no dev split, so:
  - `vidore/vidore_v3_hr` (1110 pages, 318 English queries) is the development set.
    All choices and tuning happen here.
  - `vidore/vidore_v3_finance_en` (2942 pages, 309 English queries) is the held-out
    test set. It is used once, in phase 9, with every setting frozen from hr.
- **One visual model throughout: `vidore/colqwen2.5-v0.2`.** Chosen because it is
  a standard, widely reported model (published ViDoRe v3 nDCG@10 about 0.52), so the
  visual route is a credible component and our numbers can be checked against
  published ones. Smaller models (colSmol-500M about 0.34) score below our own BM25
  on hr (0.50) and would not be a fair test of a visual route.
  - Pages and the dataset's queries are encoded once on a free cloud GPU (Kaggle
    notebook `notebooks/encode_colqwen.ipynb`, fp16), saved as `pages.npz` /
    `queries.npz`, and copied to `data/vectors/<subset>/`.
  - Queries that only exist at run time (agent rewrites, HyDE) are encoded locally
    by the same model on CPU in bfloat16 (16 GB RAM; measure the latency).
  - Check once that cloud fp16 and local bf16 vectors agree (cosine > 0.99 on a few
    pages and queries); record the result. Result: queries agree (mean cosine
    0.999, same top-10 pages 98.5% of the time on hr); pages do not (0.92–0.98).
    Pages are never encoded locally, so only the query result matters; see
    `DECISIONS.md`. On this machine the local encoder takes 5–10 s per question and
    about 33 s per passage-length text (the model does not fit in free RAM).
- English queries, seed 0, all queries of the subset for every retrieval and answer
  run.
- Break every result down by `query_types` and by number of relevant pages
  (1, 2–4, 5–9, 10+), in addition to the mean.

## Phase 5: baseline and multi-route retrieval

- Step 0: the baseline, retrieval and answers.
- Reference runs (retrieval only, not ladder steps): BM25 alone, visual alone.
- Ladder candidates, in this order:
  1. text + BM25, fused with RRF.
  2. text + BM25 + visual (every page in the visual index), fused with RRF.
  3. Per-page routing (efficiency change): the page analyzer sends each page to the
     text or the visual index, BM25 still covers every page. Compare with step 2 on
     quality and on index storage.
- Report per run: NDCG@10, Recall@10, answer score (when run), citation validity,
  index storage per page, cost, latency.

## Phase 6: retrieval improvements

### 6.1 Reranking
- Why: first-stage routes score query and page separately; a second stage can read
  them together.
- How: add a cross-encoder reranker (`BAAI/bge-reranker-base`, already in the local
  Hugging Face cache, CPU, text pages only; pages with no text keep their position)
  next to the existing `maxsim` and `llm` rerankers, same
  `rerank(query, candidates, top_n)` contract, top 20 only.
- Ladder: compare the three on hr; the best one is the candidate step. Report
  latency and cost of each.

### 6.2 Contextual chunk headers
- Why: a chunk may not say which report or section it came from.
- How (Anthropic's Contextual Retrieval): an LLM writes one short sentence that
  places each chunk in its document; the sentence is prepended before embedding and
  BM25 indexing.
- Trade-off: one cheap LLM call per chunk at index time (about 1,200 on hr).
- Ladder candidate; report indexing cost.

### 6.3 Index compression (efficiency)
- Why: index size is the main running cost of the system.
- How, both without new API calls or re-encoding:
  - text: Matryoshka truncation. `text-embedding-3-*` is trained so that a prefix of
    the vector is itself a usable embedding; truncate stored vectors to
    1536 / 768 / 512 / 256 / 128 dims and re-normalize.
  - visual: the existing quantization (float16 / int8 / binary) and token pooling
    (factor 1 / 2 / 3) of the ColQwen2.5 vectors, with and without the full-precision
    MaxSim rerank.
- Report one storage-vs-quality chart for both; keep the smallest setting whose
  quality is within noise of uncompressed.

### 6.4 Query rewriting / HyDE
- Why: users' wording can differ from the documents' wording.
- How: generate a short hypothetical answer passage and search with it, as an
  option of the one-shot pipeline. The agent (phase 7) reuses the same function.
  The text and BM25 routes search with question + passage; the visual route keeps
  the question (encoding 318 passages locally would take about three hours).
- Trade-off: one extra LLM call; the rewrite can drift from the question.
- Ladder candidate.

## Phase 7: agentic retrieval

### 7.1 Retrieval agent
- Why: some questions may need more than one retrieval (multi-hop, compare,
  enumerate, many relevant pages). Whether an agent helps is what this phase tests.
- How:
  - Tools (OpenAI function calling): `text_search`, `keyword_search`,
    `visual_search` (each returns page ids, scores and short snippets),
    `read_page` (returns the page image or text), `finish`.
  - Loop: the model decides what to search, reads results, judges whether the
    evidence is sufficient, searches again if not, then picks the pages to answer
    from (dynamic top-k instead of a fixed 5).
  - Hard limits: max tool calls (e.g. 8) and a token budget per question.
  - `visual_search` needs the local query encoder (`--local-query-encoder`); without
    it the tool is not offered. Tested as its own ladder step on hr (step 9): the
    answer score did not change (+0.000, p = 1.0) and the agent seldom used the
    tool, so the final system is the agent without it.
  - Tool-calling turns run with `reasoning_effort="none"`: the chat completions
    endpoint does not accept function tools together with reasoning for this model.
  - Reuse the existing routes, fusion, context assembly, generation and citation
    check. The agent is a new search strategy, not a second pipeline.
- Retrieval output for evaluation: the pages the agent selected, in its order,
  followed by the remaining fused candidates.
- Trade-off: several times the calls and latency; runs are less reproducible (fix
  temperature and seed where the API allows; log every step).
- Ladder candidate, judged on answer score against the current best one-shot system
  at the same final page budget; also report per query type, tool calls, tokens,
  latency and cost per question.

### 7.2 Tracing
- Why: agent runs have many steps; failures and cost need to be inspectable.
- How: OpenTelemetry tracing through OpenInference instrumentation, viewed in
  Arize Phoenix. Optional dependency group; off unless enabled.
- Deliverable: a screenshot of one traced agent run for the README. Not a ladder step.

## Phase 8: robustness checks (side evaluations, not ladder steps)

Run on the final system.

### 8.1 Abstention
- Why: ViDoRe has no unanswerable questions, so behaviour on missing evidence is
  never tested.
- How: for a sample of queries, remove every relevant page from the context and
  ask anyway.
- Measure: refusal rate on these, and false-refusal rate on normal queries.

### 8.2 Indirect prompt injection
- Why: retrieved documents are untrusted input.
- How: insert an instruction into the text (and/or image) of one relevant page,
  e.g. telling the model to answer something else. Then add one defence (clear
  delimiting of page content plus a system rule that page content is data, not
  instructions).
- Measure: attack success rate before and after the defence, on about 30 queries;
  answer score on clean queries must not drop beyond noise.

## Phase 9: held-out test and report

- Freeze the final system and every setting chosen on hr.
- On finance_en run: the baseline and the final system (retrieval and answers), the
  leave-one-out ablations of the final system (retrieval metrics only; the agent's
  effect on answers is measured on hr), and the storage comparison of routing. Do not tune anything on finance_en; if a result looks wrong, fix bugs
  only, and record it.
- `README.md` for a reader who has never seen the project: what it does, the
  architecture diagram, how to run, the ladder table (dev and held-out), the
  storage-vs-quality chart, robustness results, and the changes that did not help.
- Every number in the README links to its `results/` directory.

## Not planned (and why)

- GraphRAG / knowledge graph: ViDoRe has no corpus-wide questions, so its value
  cannot be measured here, and it costs LLM calls per chunk.
- Fine-tuning the embedding or visual model: 4 GB of VRAM is not enough.
- Late chunking: needs token-level embeddings from a long-context model; the
  OpenAI API does not return them.
- Comparing several visual models: the study compares configurations under one
  standard model.
- Approximate nearest-neighbour search (HNSW / vector database): at 1–3k pages
  brute-force MaxSim is fast, and a second store backend would be a second
  implementation of the same thing.
