"""RAG Sprints dashboard: the eval harness and the 12 architectures, read straight from the harness run files.

uv run dashboard/app.py        # then open http://127.0.0.1:7860
"""

import html
from datetime import datetime
from types import SimpleNamespace

import gradio as gr
import plotly.graph_objects as go

import data
from data import METRIC_BY_KEY, METRICS, SPRINTS, fmt

# ---------- Visual tokens (one blue for every architecture, gray for the baseline, one blue ramp) ----------

# Each mode is its own set of steps, not an inverted copy. In dark mode the ramp runs from the surface
# (missed) up to light blue (rank 1), so "more" is still "stronger".
PALETTES = {
    "light": SimpleNamespace(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        series="#2a78d6", baseline="#b9b7ae", hover_bg="#ffffff",
        ramp=[[0.0, "#f0efec"], [0.01, "#cde2fb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"], [0.75, "#256abf"], [1.0, "#0d366b"]],
    ),
    "dark": SimpleNamespace(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        series="#3987e5", baseline="#6f6d67", hover_bg="#222220",
        ramp=[[0.0, "#2c2c2a"], [0.01, "#184f95"], [0.25, "#256abf"], [0.5, "#3987e5"], [0.75, "#6da7ec"], [1.0, "#b7d3f6"]],
    ),
}


def pal(mode: str) -> SimpleNamespace:
    return PALETTES.get(mode, PALETTES["light"])


FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

TYPE_LABELS = {"simple": "Simple", "multi_part": "Multi-part", "exact_term": "Exact term", "no_answer": "No answer"}
ANSWERABLE_TYPES = ["simple", "multi_part", "exact_term"]


def style(fig: go.Figure, height: int, p: SimpleNamespace) -> go.Figure:
    fig.update_layout(
        template="none", height=height, paper_bgcolor=p.surface, plot_bgcolor=p.surface,
        font=dict(family=FONT, size=12, color=p.ink2), margin=dict(l=8, r=24, t=16, b=8),
        hoverlabel=dict(bgcolor=p.hover_bg, bordercolor=p.axis, font=dict(family=FONT, color=p.ink, size=12)),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=p.grid, linecolor=p.axis, zeroline=False, tickfont=dict(color=p.muted), automargin=True)
    fig.update_yaxes(gridcolor=p.grid, linecolor=p.axis, zeroline=False, tickfont=dict(color=p.ink2), automargin=True)
    return fig


def label(s: data.Sprint) -> str:
    return f"{s.num} {s.name}"


def baseline() -> dict:
    return data.summary(data.BASELINE_RUN)


def rows() -> list[tuple[data.Sprint, dict]]:
    return [(s, data.summary(s.run, s.cost_override)) for s in data.scored_sprints()]


def best_value(metric: data.Metric, values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return max(vals) if metric.higher_is_better else min(vals)


def delta_html(metric: data.Metric, value: float | None, base: float | None) -> str:
    if value is None or base is None:
        return ""
    diff = value - base
    if fmt(metric, abs(diff)) == fmt(metric, 0.0):  # too small to show at this precision
        return '<span class="delta flat">= baseline</span>'
    better = diff > 0 if metric.higher_is_better else diff < 0
    arrow = "▲" if diff > 0 else "▼"
    text = fmt(metric, abs(diff)) if metric.kind != "count" else f"{abs(diff):.0f}"
    return f'<span class="delta {"good" if better else "bad"}">{arrow} {text}</span>'


# ---------- Leaderboard ----------

def kpi_tiles() -> str:
    data_rows = rows()
    tiles = []
    for key, title in [("evidence_recall", "Best evidence recall"), ("evidence_mrr", "Best evidence MRR"),
                       ("faithfulness", "Best faithfulness"), ("latency_p50_s", "Fastest p50"),
                       ("cost_per_query_usd", "Cheapest per query")]:
        m = METRIC_BY_KEY[key]
        candidates = [(s, r[key]) for s, r in data_rows if r[key] is not None and s.key != "naive"]
        s, v = (max if m.higher_is_better else min)(candidates, key=lambda x: x[1])
        tiles.append(f'<div class="kpi"><div class="kpi-label">{title}</div><div class="kpi-value">{fmt(m, v)}</div>'
                     f'<div class="kpi-sub">{html.escape(label(s))} · baseline {fmt(m, baseline()[key])}</div></div>')
    return f'<div class="kpi-grid">{"".join(tiles)}</div>'


def leaderboard_table() -> str:
    data_rows = rows()
    base = baseline()
    best = {m.key: best_value(m, [r[m.key] for _, r in data_rows]) for m in METRICS}
    head = "".join(f'<th title="{"higher" if m.higher_is_better else "lower"} is better">{m.label}</th>' for m in METRICS)
    body = []
    for s, r in data_rows:
        cells = []
        for m in METRICS:
            v = r[m.key]
            is_best = v is not None and abs(v - best[m.key]) < 1e-9
            shown = f"{fmt(m, v)} / {data.n_answerable()}" if m.kind == "count" else fmt(m, v)
            d = "" if s.key == "naive" else delta_html(m, v, base[m.key])
            cells.append(f'<td class="{"best" if is_best else ""}"><div>{shown}</div>{d}</td>')
        note = ' <span class="tag">real cost</span>' if s.cost_override else ""
        row_cls = "baseline" if s.key == "naive" else ""
        body.append(f'<tr class="{row_cls}"><th class="arch"><span class="num">{s.num}</span>{html.escape(s.name)}{note}</th>'
                    f'{"".join(cells)}</tr>')
    return (f'<div class="table-wrap"><table class="lb"><thead><tr><th class="arch">Architecture</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>'
            '<p class="foot">Medical golden set v1: 30 questions (26 answerable, 4 no-answer). Bold = best in column. '
            'Arrows compare with the naive baseline (green = better). Every architecture refuses all 4 no-answer questions. '
            'Speculative shows its real cost (gpt-4o-mini drafters + gpt-4o verifier); the harness prices every token as gpt-4o-mini. '
            'Multimodal (11) runs on its own image questions, see the Architectures tab.</p>')


def metric_bar(metric_key: str, mode: str = "light") -> go.Figure:
    m, p = METRIC_BY_KEY[metric_key], pal(mode)
    data_rows = sorted(rows(), key=lambda x: x[1][m.key] if x[1][m.key] is not None else 0, reverse=m.higher_is_better)
    names = [label(s) for s, _ in data_rows]
    values = [r[m.key] for _, r in data_rows]
    best = best_value(m, values)
    colors = [p.baseline if s.key == "naive" else p.series for s, _ in data_rows]
    text = [fmt(m, v) if (s.key == "naive" or abs(v - best) < 1e-9) else "" for (s, _), v in zip(data_rows, values)]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=text, textposition="outside", textfont=dict(color=p.ink2, size=12), cliponaxis=False,
        customdata=[fmt(m, v) for v in values], hovertemplate="<b>%{y}</b><br>" + m.label + ": %{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(bargap=0.35)
    if m.kind == "score":
        fig.update_xaxes(range=[0, 1.08])
    return style(fig, 430, p)


# Points that sit close together on the default axes get their labels pushed apart
LABEL_SIDE = {"contextual": "middle left", "graph_rag": "middle right", "raptor": "middle left", "hybrid": "middle right"}


def tradeoff_scatter(quality_key: str, cost_key: str, mode: str = "light") -> go.Figure:
    q, c, p = METRIC_BY_KEY[quality_key], METRIC_BY_KEY[cost_key], pal(mode)
    data_rows = rows()
    fig = go.Figure()
    for is_base in (False, True):
        pts = [(s, r) for s, r in data_rows if (s.key == "naive") == is_base]
        fig.add_trace(go.Scatter(
            x=[r[c.key] for _, r in pts], y=[r[q.key] for _, r in pts], mode="markers+text",
            text=[s.name for s, _ in pts], textposition=[LABEL_SIDE.get(s.key, "top center") for s, _ in pts],
            textfont=dict(size=11, color=p.ink2),
            marker=dict(size=11, color=p.baseline if is_base else p.series, line=dict(width=2, color=p.surface)),
            customdata=[[label(s), fmt(q, r[q.key]), fmt(c, r[c.key])] for s, r in pts],
            hovertemplate="<b>%{customdata[0]}</b><br>" + q.label + ": %{customdata[1]}<br>" + c.label
                          + ": %{customdata[2]}<extra></extra>",
        ))
    fig.update_xaxes(title=dict(text=f"{c.label} → (lower is better)", font=dict(color=p.muted, size=12)))
    fig.update_yaxes(title=dict(text=f"{q.label} ↑", font=dict(color=p.muted, size=12)))
    if c.kind == "usd":
        fig.update_xaxes(tickformat="$.4f")
    return style(fig, 430, p)


def type_heatmap(metric_key: str, mode: str = "light") -> go.Figure:
    m, p = METRIC_BY_KEY[metric_key], pal(mode)
    sprints = data.scored_sprints()
    z, text = [], []
    for s in sprints:
        qs = data.questions(s.run)
        row, row_text = [], []
        for t in ANSWERABLE_TYPES:
            vals = [qs[g["id"]]["scores"].get(m.key) for g in data.golden() if g["question_type"] == t and g["id"] in qs]
            vals = [v for v in vals if v is not None]
            avg = sum(vals) / len(vals) if vals else None
            row.append(avg)
            row_text.append(f"{avg:.2f}" if avg is not None else "—")
        z.append(row)
        text.append(row_text)
    counts = {t: sum(1 for g in data.golden() if g["question_type"] == t) for t in ANSWERABLE_TYPES}
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{TYPE_LABELS[t]} ({counts[t]})" for t in ANSWERABLE_TYPES], y=[label(s) for s in sprints],
        text=text, colorscale=p.ramp, zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=p.muted)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>" + m.label + ": %{text}<extra></extra>",
    ))
    # One text colour can't sit on both ends of the ramp, so each cell picks dark or white ink by its own shade
    x_labels, y_labels = fig.data[0].x, fig.data[0].y
    for i, row in enumerate(z):
        for j, v in enumerate(row):
            light_cell = v is not None and (v < 0.5 if mode == "light" else v > 0.65)
            fig.add_annotation(x=x_labels[j], y=y_labels[i], text=text[i][j], showarrow=False,
                               font=dict(size=12, color="#0b0b0b" if light_cell else "#ffffff"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return style(fig, 460, p)


def question_grid(mode: str = "light") -> go.Figure:
    p = pal(mode)
    sprints = data.scored_sprints()
    answerable = [g for g in data.golden() if g["question_type"] != "no_answer"]
    z, hover = [], []
    for s in sprints:
        qs = data.questions(s.run)
        z_row, h_row = [], []
        for g in answerable:
            sc = qs[g["id"]]["scores"] if g["id"] in qs else {}
            rank = data.evidence_rank(sc)
            z_row.append(sc.get("evidence_mrr", 0) or 0)
            h_row.append(f"{g['id']} · {TYPE_LABELS[g['question_type']]}<br>{html.escape(g['question'][:90])}<br>"
                         + (f"evidence at rank {rank}" if rank else "evidence not retrieved"))
        z.append(z_row)
        hover.append(h_row)
    fig = go.Figure(go.Heatmap(
        z=z, x=[g["id"] for g in answerable], y=[label(s) for s in sprints], customdata=hover,
        colorscale=p.ramp, zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=p.muted), tickvals=[0, 0.2, 0.5, 1],
                      ticktext=["missed", "rank 5", "rank 2", "rank 1"]),
        hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False, tickangle=-45, tickfont=dict(size=10))
    return style(fig, 460, p)


# ---------- Architecture cards ----------

def sprint_choices() -> list[tuple[str, str]]:
    return [(label(s), s.key) for s in SPRINTS]


SIDE_SECTIONS = ["When I'd reach for it", "Fusion tuning", "What it gets wrong"]


def arch_card(key: str, mode: str = "light") -> tuple[str, go.Figure | None, str, str]:
    s = data.sprint_by_key(key)
    intro = html.escape(data.readme_intro(s.folder))
    diagram = html.escape(data.readme_diagram(s.folder))
    run_line = f'run <code>{s.run}</code>' if s.run else "not on the golden set"
    head = (f'<div class="card"><div class="card-num">Sprint {s.num}</div><h2>{html.escape(s.name)}</h2>'
            f'<p class="lede">{intro}</p><p class="meta"><code>rag_sprints/{s.folder}</code> · {run_line}</p>'
            + (f'<pre class="diagram">{diagram}</pre>' if diagram else "") + "</div>")
    if s.run:
        r, base = data.summary(s.run, s.cost_override), baseline()
        tiles = []
        for m in METRICS:
            shown = f"{fmt(m, r[m.key])} / {data.n_answerable()}" if m.kind == "count" else fmt(m, r[m.key])
            d = "" if s.key == "naive" else delta_html(m, r[m.key], base[m.key])
            tiles.append(f'<div class="kpi small"><div class="kpi-label">{m.label}</div>'
                         f'<div class="kpi-value">{shown}</div>{d}</div>')
        head += f'<div class="kpi-grid five">{"".join(tiles)}</div>'
        fig = arch_type_bars(s, pal(mode))
    else:
        head += ('<div class="notice">This sprint is not scored on the golden set: the corpus is text, so pictures '
                 'cannot change its scores. It runs on its own 6 synthetic images and 7 questions; '
                 'its results are below.</div>')
        fig = None
    # Early sprints predate the "When I'd reach for it" section; fall back to their closest one
    side_heading, side_body = next(((h, data.readme_section(s.folder, h)) for h in SIDE_SECTIONS
                                    if data.readme_section(s.folder, h)), ("", ""))
    results_md = data.readme_section(s.folder, "Results (7 sprint-local questions)") if not s.run else ""
    side_md = (results_md + "\n\n" if results_md else "") + (f"### {side_heading}\n\n{side_body}" if side_body else "")
    return head, fig, side_md, data.readme(s.folder)


def arch_type_bars(s: data.Sprint, p: SimpleNamespace) -> go.Figure:
    """This architecture vs the baseline, evidence recall per question type."""
    fig = go.Figure()
    for run, name, color in [(data.BASELINE_RUN, "Naive baseline", p.baseline), (s.run, s.name, p.series)]:
        qs = data.questions(run)
        ys = []
        for t in ANSWERABLE_TYPES:
            vals = [qs[g["id"]]["scores"].get("evidence_recall", 0) for g in data.golden()
                    if g["question_type"] == t and g["id"] in qs]
            ys.append(sum(vals) / len(vals) if vals else 0)
        fig.add_trace(go.Bar(x=[TYPE_LABELS[t] for t in ANSWERABLE_TYPES], y=ys, name=name,
                             marker=dict(color=color, cornerradius=4),
                             hovertemplate=f"<b>{html.escape(name)}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.05], title=dict(text="Evidence recall", font=dict(color=p.muted)))
    fig = style(fig, 300, p)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0, font=dict(color=p.ink2)))
    return fig


# ---------- Question explorer ----------

def question_choices() -> list[tuple[str, str]]:
    return [(f"{g['id']} · {TYPE_LABELS[g['question_type']]} · {g['question']}", g["id"]) for g in data.golden()]


def question_view(gid: str) -> str:
    g = data.golden_by_id()[gid]
    no_answer = g["question_type"] == "no_answer"
    head = (f'<div class="card"><span class="badge">{TYPE_LABELS[g["question_type"]]}</span>'
            f'<span class="badge ghost">{g["id"]} · {html.escape(g.get("source_doc") or "no source")}</span>'
            f'<h3>{html.escape(g["question"])}</h3>'
            f'<p class="meta"><b>Expected:</b> {html.escape(g["ground_truth"])}</p>'
            + (f'<blockquote>{html.escape(g["evidence"])}</blockquote>' if g.get("evidence") else "") + "</div>")
    body = []
    for s in data.scored_sprints():
        q = data.questions(s.run).get(gid)
        if not q:
            continue
        sc, out = q["scores"], q["output"] or {}
        refused = bool(sc.get("refused"))
        if no_answer:
            verdict = ('<span class="pill good">✓ refused</span>' if refused
                       else '<span class="pill bad">✗ answered</span>')
        else:
            rank = data.evidence_rank(sc)
            verdict = (f'<span class="pill good">✓ evidence #{rank}</span>' if rank
                       else '<span class="pill bad">✗ evidence missed</span>')
            if refused:
                verdict += ' <span class="pill bad">refused</span>'
        faith = sc.get("faithfulness")
        faith_txt = f"{faith:.2f}" if faith is not None else "—"
        answer = html.escape(out.get("answer") or (q.get("error") or "")).replace("\n", "<br>")
        body.append(f'<tr><th class="arch"><span class="num">{s.num}</span>{html.escape(s.name)}</th>'
                    f'<td>{verdict}</td><td class="numcell">{faith_txt}</td>'
                    f'<td class="numcell">{q["latency_ms"] / 1000:.1f} s</td><td class="answer">{answer}</td></tr>')
    table = ('<div class="table-wrap"><table class="lb qx"><thead><tr><th class="arch">Architecture</th>'
             f'<th>{"Refusal" if no_answer else "Evidence"}</th><th>Faithful.</th><th>Latency</th><th>Answer</th>'
             f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return head + table


# ---------- Compare runs ----------

def run_choices() -> list[tuple[str, str]]:
    out = []
    for f in data.list_runs():
        run = data.load_run(f)
        when = datetime.fromisoformat(run["started_at"]).strftime("%d %b %H:%M")
        out.append((f"{run['arch']} · {when} · {f}", f))
    return out


def compare_view(a: str, b: str) -> str:
    if not a or not b:
        return ""
    ra, rb = data.summary(a), data.summary(b)
    rows_html = []
    for m in METRICS:
        rows_html.append(f'<tr><th class="arch">{m.label}</th><td class="numcell">{fmt(m, ra[m.key])}</td>'
                         f'<td class="numcell">{fmt(m, rb[m.key])}</td><td>{delta_html(m, rb[m.key], ra[m.key])}</td></tr>')
    metrics_table = ('<div class="table-wrap narrow"><table class="lb narrow"><thead><tr><th class="arch">Metric</th>'
                     f'<th>A · {html.escape(data.load_run(a)["arch"])}</th><th>B · {html.escape(data.load_run(b)["arch"])}</th>'
                     f'<th>B vs A</th></tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>')
    qa, qb = data.questions(a), data.questions(b)
    changed = []
    for g in data.golden():
        if g["id"] not in qa or g["id"] not in qb:
            continue
        sa, sb = qa[g["id"]]["scores"], qb[g["id"]]["scores"]
        ra_, rb_ = data.evidence_rank(sa), data.evidence_rank(sb)
        fa, fb = sa.get("faithfulness"), sb.get("faithfulness")
        if ra_ == rb_ and bool(sa.get("refused")) == bool(sb.get("refused")) and \
                (fa is None or fb is None or abs(fa - fb) < 0.3):
            continue

        def cell(rank, refused, faith):
            parts = [f"#{rank}" if rank else "missed"]
            if refused:
                parts.append("refused")
            if faith is not None:
                parts.append(f"faith {faith:.2f}")
            return " · ".join(parts)

        changed.append(f'<tr><th class="arch">{g["id"]}</th><td class="q">{html.escape(g["question"])}</td>'
                       f'<td>{cell(ra_, sa.get("refused"), fa)}</td><td>{cell(rb_, sb.get("refused"), fb)}</td></tr>')
    q_table = ('<h3 class="section">Questions that changed</h3><p class="foot">Evidence rank, refusal, or faithfulness '
               'moved by 0.3 or more.</p>' + (
        '<div class="table-wrap"><table class="lb qx"><thead><tr><th class="arch">ID</th><th>Question</th><th>A</th><th>B</th>'
        f'</tr></thead><tbody>{"".join(changed)}</tbody></table></div>' if changed
        else '<p class="foot">No question changed.</p>'))
    return metrics_table + q_table


# ---------- Harness ----------

def harness_overview() -> str:
    counts = {t: sum(1 for g in data.golden() if g["question_type"] == t) for t in TYPE_LABELS}
    tiles = "".join(f'<div class="kpi small"><div class="kpi-label">{TYPE_LABELS[t]}</div>'
                    f'<div class="kpi-value">{n}</div></div>' for t, n in counts.items())
    runs = data.list_runs()
    run_rows = []
    for f in runs:
        r = data.load_run(f)
        s = r["summary"]
        when = datetime.fromisoformat(r["started_at"]).strftime("%d %b %Y %H:%M")
        run_rows.append(f'<tr><th class="arch">{html.escape(r["arch"])}</th><td><code>{f}</code></td><td>{when}</td>'
                        f'<td class="numcell">{s["evidence_recall"]:.2f}</td><td class="numcell">{s["evidence_mrr"]:.2f}</td>'
                        f'<td class="numcell">${s["cost_per_query_usd"]:.5f}</td></tr>')
    return (f'<div class="kpi-grid four">{tiles}</div>'
            '<h3 class="section">Every run on the medical golden set</h3>'
            '<div class="table-wrap"><table class="lb qx"><thead><tr><th class="arch">Arch</th><th>File</th><th>Started</th>'
            f'<th>Evid. recall</th><th>Evid. MRR</th><th>$/query (harness)</th></tr></thead><tbody>{"".join(run_rows)}'
            '</tbody></table></div>')


HARNESS_MD = """
### The contract

Every architecture is a function from a question to a `PipelineOutput`. The harness runs it on every golden question, times it (the runner measures latency, not the pipeline), scores it and saves `eval_harness/results/<arch>_<time>.json`.

```python
from eval_harness import PipelineOutput, evaluate

def my_pipeline(question: str) -> PipelineOutput:
    return PipelineOutput(
        answer="...",
        contexts=["chunk text the LLM read", ...],   # rank order
        context_ids=["hvmc_formulary", ...],          # source doc of each chunk
        tokens_in=850, tokens_out=20,                 # all LLM calls
    )

evaluate(my_pipeline, arch="hybrid", golden_set="golden_medical_v1.jsonl")
```

### What it measures

| Family | Metrics | Cost |
|---|---|---|
| Retrieval | Evidence recall and MRR: did the exact evidence quote come back, and at which rank? | free |
| Generation | RAGAS faithfulness, answer relevancy, context precision, context recall | gpt-4o-mini judge |
| Refusals | Abstention on no-answer questions · refused although answerable | judge · free |
| Engineering | Latency p50 / p95, cost per query, error rate | free |

Refusals are not scored for faithfulness: "I don't know" makes no claims, so there is nothing to check. They are counted as *refused although answerable* instead.

### The golden set

`golden_medical_v1.jsonl`: 30 questions over a synthetic, fictional hospital (formulary, lab catalog, protocols, device manuals, leaflets). Built to break retrieval: near-identical codes (`HV-RX-2047` vs `HV-RX-2074`), fault codes reused across manuals, look-alike drug names, plain-language questions over technical text. Every answerable question carries a verbatim evidence quote.
"""


# ---------- Layout ----------

CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --surface-2: #f6f6f3; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hair: #e1e0d9; --border: rgba(11,11,11,.08); --series: #2a78d6; --row-hover: #f4f7fd;
  --good: #006300; --bad: #d03b3b; --good-bg: #eaf5ea; --bad-bg: #fbeceb; --bad-ink: #b02a2a;
  --notice-bg: #f4f7fd; --notice-border: #cde2fb;
}
.dark {
  --page: #0d0d0d; --surface: #1a1a19; --surface-2: #222220; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --hair: #2c2c2a; --border: rgba(255,255,255,.10); --series: #3987e5; --row-hover: rgba(57,135,229,.10);
  --good: #0ca30c; --bad: #e66767; --good-bg: rgba(12,163,12,.14); --bad-bg: rgba(230,103,103,.14); --bad-ink: #e66767;
  --notice-bg: rgba(57,135,229,.10); --notice-border: rgba(57,135,229,.35);
}
body, .gradio-container { background: var(--page) !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.hero { padding: 28px 4px 8px; }
.hero .eyebrow { font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--series); font-weight: 600; }
.hero h1 { font-size: 30px; line-height: 1.15; margin: 6px 0 8px; color: var(--ink); font-weight: 650; }
.hero p { color: var(--ink-2); font-size: 15px; max-width: 820px; margin: 0; }
.theme-btn { margin-top: 30px !important; }
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 8px 0 4px; }
.kpi-grid.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.kpi.small { padding: 10px 14px; }
.kpi-label { font-size: 12px; color: var(--muted); }
.kpi-value { font-size: 24px; font-weight: 650; color: var(--ink); margin-top: 2px; }
.kpi.small .kpi-value { font-size: 19px; }
.kpi-sub { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
.table-wrap.narrow { max-width: 760px; }
table.lb { border: 0 !important; border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; background: transparent !important; }
table.lb th, table.lb td { padding: 9px 12px; border-bottom: 1px solid var(--hair) !important; text-align: right; vertical-align: top; color: var(--ink); background: transparent; }
.table-wrap table, .table-wrap thead, .table-wrap tbody, .table-wrap tr, .table-wrap th, .table-wrap td { border-left: none !important; border-right: none !important; border-top: none !important; box-shadow: none !important; outline: none !important; }
table.lb tbody th, table.lb tbody td { background: transparent !important; }
table.lb thead th { font-size: 11.5px; font-weight: 600; color: var(--muted); background: var(--surface-2) !important; position: sticky; top: 0; }
table.lb th.arch { text-align: left; font-weight: 600; white-space: nowrap; }
table.lb tbody tr { background: transparent !important; }
table.lb tbody tr:hover { background: var(--row-hover) !important; }
table.lb tbody tr.baseline { background: var(--surface-2) !important; }
table.lb td.best div { font-weight: 700; }
table.lb td.answer, table.lb td.q { text-align: left; color: var(--ink-2); min-width: 320px; }
table.lb.qx td, table.lb.qx thead th { text-align: left; } table.lb td.numcell { text-align: right; white-space: nowrap; }
table.lb.narrow { max-width: 760px; }
table.lb code, .meta code { color: var(--ink-2); background: transparent; }
.num { display: inline-block; min-width: 24px; color: var(--muted); font-weight: 500; }
.delta { display: block; font-size: 11px; margin-top: 1px; }
.delta.good { color: var(--good); } .delta.bad { color: var(--bad); } .delta.flat { color: var(--muted); }
.tag { font-size: 10px; font-weight: 600; color: var(--ink-2); border: 1px solid var(--hair); border-radius: 6px; padding: 1px 5px; margin-left: 6px; }
.foot { font-size: 12px; color: var(--muted); margin: 8px 2px 0; line-height: 1.5; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; }
.card h2 { margin: 2px 0 6px; font-size: 22px; color: var(--ink); } .card h3 { margin: 10px 0 6px; font-size: 18px; color: var(--ink); }
.card-num { font-size: 12px; color: var(--series); font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.lede { font-size: 15px; color: var(--ink-2); margin: 0 0 6px; }
.meta { font-size: 13px; color: var(--ink-2); margin: 4px 0; }
.meta b { color: var(--ink); }
.diagram { background: var(--surface-2); border-radius: 10px; padding: 12px 14px; font-size: 12px; line-height: 1.45; overflow-x: auto; color: var(--ink); margin-top: 10px; }
blockquote { border-left: 3px solid var(--series); margin: 8px 0 0; padding: 4px 12px; color: var(--ink-2); font-size: 13px; background: var(--surface-2); border-radius: 0 8px 8px 0; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; color: #fff; background: var(--series); border-radius: 999px; padding: 2px 9px; margin-right: 6px; }
.badge.ghost { background: transparent; color: var(--ink-2); border: 1px solid var(--hair); }
.pill { display: inline-block; font-size: 11.5px; font-weight: 600; border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
.pill.good { color: var(--good); background: var(--good-bg); } .pill.bad { color: var(--bad-ink); background: var(--bad-bg); }
.notice { background: var(--notice-bg); border: 1px solid var(--notice-border); color: var(--ink-2); border-radius: 12px; padding: 12px 16px; font-size: 14px; }
h3.section { font-size: 15px; color: var(--ink); margin: 18px 2px 4px; }
.panel-title { font-size: 15px; font-weight: 600; color: var(--ink); margin: 6px 2px 0; }
.panel-sub { font-size: 12.5px; color: var(--muted); margin: 0 2px 4px; }
.prose table, .prose th, .prose td, .md table, .md th, .md td { border-color: var(--hair) !important; }
.prose th, .md th { background: var(--surface-2) !important; }
@media (max-width: 900px) { .kpi-grid, .kpi-grid.four { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hero h1 { font-size: 24px; } }
"""

# Gradio marks dark mode with a "dark" class (from the OS setting or ?__theme=dark). Charts are drawn in Python,
# so the page tells Python which mode it is in on load and on every toggle, and the charts are redrawn.
IS_DARK_JS = "const isDark = () => !!document.querySelector('.dark');"
LOAD_JS = f"(m, ...rest) => {{ {IS_DARK_JS} return [isDark() ? 'dark' : 'light', ...rest]; }}"
TOGGLE_JS = f"""(m, ...rest) => {{ {IS_DARK_JS}
  const dark = !isDark();
  document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
  if (dark) document.body.classList.add('dark');
  return [dark ? 'dark' : 'light', ...rest]; }}"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.Font(f) for f in ("system-ui", "-apple-system", "Segoe UI", "sans-serif")], radius_size=gr.themes.sizes.radius_md,
).set(body_background_fill="#f9f9f7", body_background_fill_dark="#0d0d0d",
      block_background_fill="#fcfcfb", block_background_fill_dark="#1a1a19",
      block_border_width="0px", block_shadow="none", block_shadow_dark="none",
      button_primary_background_fill="#2a78d6", button_primary_background_fill_dark="#3987e5",
      input_background_fill_dark="#222220", input_border_color_dark="#383835",
      button_secondary_background_fill_dark="#222220", button_secondary_border_color_dark="#383835",
      button_secondary_text_color_dark="#ffffff")


def panel(title: str, sub: str) -> None:
    gr.HTML(f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>')


def redraw(mode: str, bar_key: str, q_key: str, c_key: str, hm_key: str, arch_key: str):
    """Every chart in the given mode (the tables and cards follow the CSS on their own)."""
    return (mode, metric_bar(bar_key, mode), tradeoff_scatter(q_key, c_key, mode), type_heatmap(hm_key, mode),
            question_grid(mode), arch_card(arch_key, mode)[1])


def build(tab: str | None = None) -> gr.Blocks:
    score_metrics = [(m.label, m.key) for m in METRICS if m.kind == "score"]
    cost_metrics = [(m.label, m.key) for m in METRICS if m.kind in ("usd", "seconds")]
    with gr.Blocks(title="RAG Sprints") as demo:
        mode = gr.Textbox("light", visible=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=10):
                gr.HTML('<div class="hero"><div class="eyebrow">RAG Sprints · Eval Harness</div>'
                        '<h1>12 RAG architectures, one ruler</h1>'
                        '<p>Every architecture answers the same 30 questions over a synthetic hospital corpus and is scored by the same '
                        'harness: did the evidence come back, at which rank, is the answer faithful, what did it cost and how long did it take.</p></div>')
            with gr.Column(scale=1, min_width=150):
                theme_btn = gr.Button("◐  Light / Dark", size="sm", variant="secondary", elem_classes="theme-btn")
        with gr.Tabs(selected=tab):
            with gr.Tab("Leaderboard", id="leaderboard"):
                gr.HTML(kpi_tiles())
                gr.HTML(leaderboard_table())
                with gr.Row(equal_height=False):
                    with gr.Column():
                        panel("Ranking", "Gray = naive baseline. Labels on the best bar and the baseline; hover for every value.")
                        bar_metric = gr.Dropdown(choices=[(m.label, m.key) for m in METRICS], value="evidence_mrr",
                                                 show_label=False, container=False)
                        bar_plot = gr.Plot(metric_bar("evidence_mrr"), show_label=False)
                    with gr.Column():
                        panel("Quality vs price", "Up and to the left is better.")
                        with gr.Row():
                            sc_q = gr.Dropdown(choices=score_metrics, value="evidence_recall", show_label=False, container=False)
                            sc_c = gr.Dropdown(choices=cost_metrics, value="cost_per_query_usd", show_label=False, container=False)
                        sc_plot = gr.Plot(tradeoff_scatter("evidence_recall", "cost_per_query_usd"), show_label=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        panel("By question type", "Average per question type, answerable questions only.")
                        hm_metric = gr.Dropdown(choices=score_metrics, value="evidence_recall", show_label=False, container=False)
                        hm_plot = gr.Plot(type_heatmap("evidence_recall"), show_label=False)
                    with gr.Column(scale=3):
                        panel("Every answerable question", "Where the evidence chunk landed. Stronger blue = higher rank; "
                                                           "a blank cell = never retrieved.")
                        grid_plot = gr.Plot(question_grid(), show_label=False)
                bar_metric.change(metric_bar, [bar_metric, mode], bar_plot)
                sc_q.change(tradeoff_scatter, [sc_q, sc_c, mode], sc_plot)
                sc_c.change(tradeoff_scatter, [sc_q, sc_c, mode], sc_plot)
                hm_metric.change(type_heatmap, [hm_metric, mode], hm_plot)

            with gr.Tab("Architectures", id="architectures"):
                arch = gr.Dropdown(choices=sprint_choices(), value="hybrid", show_label=False, container=False)
                first = arch_card("hybrid")
                card = gr.HTML(first[0])
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        type_plot = gr.Plot(first[1], show_label=False)
                    with gr.Column(scale=2):
                        side = gr.Markdown(first[2])
                with gr.Accordion("Full sprint README", open=False):
                    full = gr.Markdown(first[3])
                arch.change(arch_card, [arch, mode], [card, type_plot, side, full])

            with gr.Tab("Question explorer", id="questions"):
                panel("One question, every architecture", "The expected answer and evidence, then what each architecture retrieved and said.")
                q_pick = gr.Dropdown(choices=question_choices(), value="m01", show_label=False, container=False)
                q_view = gr.HTML(question_view("m01"))
                q_pick.change(question_view, q_pick, q_view)

            with gr.Tab("Compare runs", id="compare"):
                panel("Any two runs, side by side", "Every run file on the medical golden set, including variants and reruns.")
                choices = run_choices()
                default_a = data.BASELINE_RUN
                default_b = data.sprint_by_key("agentic").run
                with gr.Row():
                    run_a = gr.Dropdown(choices=choices, value=default_a, label="Run A")
                    run_b = gr.Dropdown(choices=choices, value=default_b, label="Run B")
                cmp_view = gr.HTML(compare_view(default_a, default_b))
                run_a.change(compare_view, [run_a, run_b], cmp_view)
                run_b.change(compare_view, [run_a, run_b], cmp_view)

            with gr.Tab("Eval harness", id="harness"):
                gr.Markdown(HARNESS_MD)
                panel("Golden set composition", "30 questions in four types.")
                gr.HTML(harness_overview())

        chart_inputs = [mode, bar_metric, sc_q, sc_c, hm_metric, arch]
        chart_outputs = [mode, bar_plot, sc_plot, hm_plot, grid_plot, type_plot]
        theme_btn.click(redraw, chart_inputs, chart_outputs, js=TOGGLE_JS)
        demo.load(redraw, chart_inputs, chart_outputs, js=LOAD_JS)  # match the OS / URL theme on first open
    return demo


if __name__ == "__main__":
    build().launch(theme=THEME, css=CSS)
