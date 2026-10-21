# Dashboard

A local web UI for the eval harness and the 12 sprints. It reads the harness run files (`eval_harness/results/*.json`), the golden set and the sprint READMEs directly, so nothing is copied by hand and nothing costs money.

```bash
uv run dashboard/app.py        # from rag_sprints/, then open http://127.0.0.1:7860
```

| Tab | What it shows |
|---|---|
| Leaderboard | Headline tiles, the full results table with changes vs the naive baseline, a ranking per metric, quality vs cost/latency, scores by question type, and where the evidence landed for every question |
| Architectures | One card per sprint: its pitch and diagram from the README, all metrics vs baseline, evidence recall by question type, "when I'd reach for it", and the full README |
| Question explorer | One golden question: the expected answer and evidence, then every architecture's evidence rank, faithfulness, latency and answer |
| Compare runs | Any two run files side by side, including variants and reruns, and the questions that changed between them |
| Eval harness | The pipeline contract, the metrics, the golden set, and every run on it |

**Light and dark.** The page opens in your system's mode (or force one with `?__theme=dark` / `?__theme=light`), and the **◐ Light / Dark** button switches it. Charts are drawn in Python, so each mode has its own palette in `PALETTES` in `app.py`, and every chart is redrawn on a switch; tables and cards follow the CSS colour tokens.

**Adding a run.** New run files appear in *Compare runs* and *Eval harness* automatically. To make a run the one an architecture shows everywhere else, set its `run` in `SPRINTS` in `data.py`. A sprint that mixes models can set `cost_override` to its real cost per query (Speculative does), because the harness prices every token as gpt-4o-mini.

| File | Purpose |
|---|---|
| `data.py` | The sprint registry, metric definitions, and readers for run files, the golden set and READMEs |
| `app.py` | The Gradio app: charts (Plotly), tables, cards and tabs |
