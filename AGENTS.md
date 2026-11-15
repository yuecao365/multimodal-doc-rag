# Multimodal Document RAG

Question answering over PDF pages. The system retrieves the relevant pages, a vision
language model (VLM) answers from them, and every answer cites pages with quotes
that are checked by code.

Built on [RAG-Anything](https://github.com/HKUDS/RAG-Anything) / LightRAG, which
have no whole-page visual retrieval. This project adds:

1. **Whole-page visual retrieval**: page images as multi-vectors from a
   late-interaction model (ColPali family), scored with MaxSim.
2. **Per-page routing**: text-heavy pages go to text retrieval, chart/table-heavy
   pages to visual retrieval; BM25 covers every page; results are fused.

Headline result: an improvement ladder from a simple, standard text-RAG baseline to
the final system, each step measured, confirmed on a held-out subset (see
`PLAN.md`). Efficiency question alongside it: how much storage and cost per-page
routing and compression save at equal quality.

## Architecture

```
index   page analyzer → text pages   → chunk + embed (LightRAG, naive mode, no KG)
                      → visual pages → multi-vectors (pooling / quantization)
        all pages     → BM25 inverted index
query   three routes → RRF fusion → optional rerank → context assembly
        → VLM answer with citations → citation check
```

- Entry points: `src/mmrag/pipelines/factory.py` (assembly), `pipelines/hybrid.py`
  (routing, fusion, rerank), `experiments/run.py` (retrieval runs),
  `experiments/answer.py` (answer runs), `experiments/ladder.py` (the ladder table
  with its significance tests, from `results/ladder.json`).
- ViDoRe ships parsed Markdown per page, so no PDF parser (Docling/MinerU) is wired
  in, and the knowledge graph is disabled (`NoKGLLM`).
- Generation builds its own prompt rather than RAG-Anything's `vlm_enhanced` query,
  because the context is a fused page list.

## Evaluation

- Data: ViDoRe v3 (page images, parsed Markdown, human-checked answers). Default
  subset `vidore/vidore_v3_hr`, English queries.
- Retrieval: NDCG@10, Recall@k (pytrec_eval, same as vidore-benchmark).
- Answers: LLM judge against reference answers, calibrated with a human-checked
  sample (agreement, Cohen's kappa).
- Citations: page in context and quote found on the page.
- Cost: index storage per page, model calls, tokens, latency.
- Comparisons: the improvement ladder (each change kept only if a paired
  significance test on the dev set says it helps) and leave-one-out ablations of
  the final system.
- Leaderboard: `adapters/vidore_pipeline.py` implements vidore-benchmark's
  `BasePipeline`.

## Phases

1–4 are done: baseline, visual retrieval, hybrid with routing and fusion,
reranking / citation check / failure analysis.

Phases 5–9 are in `PLAN.md`. Before starting a phase, read only that phase's
section of `PLAN.md`.

`DECISIONS.md` records each choice made during implementation and its reason. It is
written in Chinese for the project owner. Add a short entry for every non-obvious
choice, and remove or update entries that are no longer true.

## Design principles

- **Simple first.** Write the smallest design that does the job of the current
  phase. No abstractions, options or extension points for hypothetical future needs.
- **Fit the existing shape.** Express new work through the interfaces that exist:
  retrievers implement `index` / `search`, rerankers implement
  `rerank(query, candidates, top_n)`, options are `PipelineSpec` fields. Add a new
  interface only when the existing ones cannot express the change, and record why.
- **Fix the design, don't patch around it.** When something does not fit, change
  the design (refactor) instead of adding special cases, flags or try/except
  workarounds. When new code replaces old code, delete the old code; never keep two
  implementations of the same thing.
- **One switch per technique.** Each technique from `PLAN.md` is a single option,
  off by default, so it can be compared with the run without it.
- **Small, clear modules.** One responsibility per module; short functions; names
  that say what they do; comments explain why, not what.
- **Few dependencies.** Add one only when it saves real work; note it in
  `DECISIONS.md`. No LangChain / LlamaIndex.
- **Keep docs true.** After a change, update `AGENTS.md`, `PLAN.md` and
  `DECISIONS.md` wherever they no longer match the code.

## Conventions

- Python 3.12 with `uv`. Code in `src/mmrag/`, tests in `tests/`. Run `ruff check`,
  `ruff format` and the full test suite before every commit.
- Tests run offline with stand-in models; no test may call a paid API.
- **Do not run anything that calls a paid API** (embeddings, VLM, judge, LLM rerank,
  agent) unless the user explicitly asks for that run. Every paid script records
  its cost in `results/spend.md` and refuses to start once the total reaches
  `BUDGET_USD`; measure a new kind of run on 20 queries first.
- Indexes and caches live under `MMRAG_WORK_ROOT` (see `.env.example`), one
  `--work-dir` per subset, so runs on the same subset share their indexes.
- Each experiment writes its config and results to `results/<date>_<name>/` and
  must be reproducible. Numbers come only from actual runs.
- The visual model never runs in the project `.venv` (colpali-engine and
  RAG-Anything need incompatible `transformers`). Pages and dataset queries are
  encoded on Kaggle (`notebooks/encode_colqwen.ipynb`) into `data/vectors/<subset>/`;
  the local visual environment `D:\rag_cache\venv_gpu` runs the agreement check and
  the run-time query encoder. Everything else runs in the project `.venv` on CPU.
- `reference/` holds read-only upstream clones (RAG-Anything, vidore-benchmark).
  Read them, never modify or commit them.
- Secrets live only in `.env`, never in code, docs or commits.
