# rag-sprints

Hands-on builds of 12 RAG architectures (plus a naive baseline). Each one runs on the same corpus and is scored by [eval-harness](https://github.com/adityajadhavv18/eval-harness).

| Week 1 | Week 2 |
|---|---|
| 00 Naive baseline | 06 Corrective RAG |
| 01 Hybrid RAG | 07 Self-RAG |
| 02 Parent-Child RAG | 08 Agentic RAG |
| 03 Contextual RAG | 09 GraphRAG |
| 04 RAG Fusion | 10 RAPTOR |
| 05 Adaptive RAG | 11 Multimodal RAG |
| | 12 Speculative RAG |

## Setup

This repo depends on `eval-harness` through a relative path, so **clone both repos side by side**:

```bash
git clone https://github.com/adityajadhavv18/eval-harness.git eval_harness
git clone https://github.com/adityajadhavv18/rag-sprints.git rag_sprints
cd rag_sprints
cp .env.example .env          # add your OpenAI and LangSmith keys
uv sync
uv run scripts/download_10ks.py   # optional: re-fetch the raw SEC filings
```

## Corpus

`data/` holds the latest 10-K filings for Apple (FY2025), Microsoft (FY2026) and Tesla (FY2025), fetched from SEC EDGAR and converted to clean text by `scripts/download_10ks.py`.

## Checks

One script per eval-harness phase, run with `uv run <script>`: `check_config.py`, `check_schema.py`, `check_golden.py`, `check_runner.py`, `check_metrics.py`.
