"""Read-only data layer for the dashboard: the sprint registry, harness run files, golden set, sprint READMEs.

Nothing here calls an LLM or writes a file. To show a new run, point its sprint's `run` at the new file.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

SPRINTS_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = SPRINTS_DIR.parent / "eval_harness"
RESULTS_DIR = HARNESS_DIR / "results"
GOLDEN_FILE = HARNESS_DIR / "golden_sets" / "golden_medical_v1.jsonl"
BASELINE_RUN = "baseline_medical_v1.json"


@dataclass(frozen=True)
class Sprint:
    num: str
    key: str
    name: str
    folder: str
    run: str | None            # harness run file in eval_harness/results, None = not on the golden set
    cost_override: float | None = None  # real $/query when a sprint mixes models (the harness prices all tokens as gpt-4o-mini)


SPRINTS = [
    Sprint("00", "naive", "Naive (baseline)", "week1/00_naive_baseline", BASELINE_RUN),
    Sprint("01", "hybrid", "Hybrid", "week1/01_hybrid_rag", "hybrid_20261011-103715.json"),
    Sprint("02", "parent_child", "Parent-Child", "week1/02_parent_child_rag", "parent_child_20261012-101557.json"),
    Sprint("03", "contextual", "Contextual", "week1/03_contextual_rag", "contextual_20261012-211033.json"),
    Sprint("04", "rag_fusion", "RAG Fusion", "week1/04_rag_fusion", "rag_fusion_20261013-170908.json"),
    Sprint("05", "adaptive", "Adaptive", "week1/05_adaptive_rag", "adaptive_20261015-134945.json"),
    Sprint("06", "corrective", "Corrective", "week2/06_corrective_rag", "corrective_20261015-202424.json"),
    Sprint("07", "self_rag", "Self-RAG", "week2/07_self_rag", "self_rag_20261017-144110.json"),
    Sprint("08", "agentic", "Agentic", "week2/08_agentic_rag", "agentic_20261017-194940.json"),
    Sprint("09", "graph_rag", "GraphRAG", "week2/09_graph_rag", "graph_rag_20261018-144833.json"),
    Sprint("10", "raptor", "RAPTOR", "week2/10_raptor", "raptor_20261018-220848.json"),
    Sprint("11", "multimodal", "Multimodal", "week2/11_multimodal_rag", None),
    Sprint("12", "speculative", "Speculative", "week2/12_speculative_rag", "speculative_20261019-214606.json",
           cost_override=0.001081),
]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    higher_is_better: bool
    kind: str  # "score" (0-1), "seconds", "usd", "count"


METRICS = [
    Metric("evidence_recall", "Evidence recall", True, "score"),
    Metric("evidence_mrr", "Evidence MRR", True, "score"),
    Metric("context_recall", "Context recall", True, "score"),
    Metric("context_precision", "Context precision", True, "score"),
    Metric("faithfulness", "Faithfulness", True, "score"),
    Metric("answer_relevancy", "Answer relevancy", True, "score"),
    Metric("false_refusals", "Refused although answerable", False, "count"),
    Metric("latency_p50_s", "Latency p50", False, "seconds"),
    Metric("latency_p95_s", "Latency p95", False, "seconds"),
    Metric("cost_per_query_usd", "Cost per query", False, "usd"),
]
METRIC_BY_KEY = {m.key: m for m in METRICS}


def fmt(metric: Metric, value: float | None) -> str:
    if value is None:
        return "—"
    if metric.kind == "score":
        return f"{value:.2f}"
    if metric.kind == "seconds":
        return f"{value:.1f} s"
    if metric.kind == "usd":
        return f"${value:.5f}"
    return f"{int(round(value))}"


@cache
def golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_FILE.read_text().splitlines() if line.strip()]


@cache
def golden_by_id() -> dict[str, dict]:
    return {g["id"]: g for g in golden()}


def n_answerable() -> int:
    return sum(1 for g in golden() if g["question_type"] != "no_answer")


@cache
def load_run(filename: str) -> dict:
    return json.loads((RESULTS_DIR / filename).read_text())


def list_runs() -> list[str]:
    """Every run file on the medical golden set, newest first."""
    runs = []
    for path in RESULTS_DIR.glob("*.json"):
        try:
            if load_run(path.name).get("golden_set") == GOLDEN_FILE.name:
                runs.append(path.name)
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(runs, key=lambda f: load_run(f).get("started_at", ""), reverse=True)


def summary(filename: str, cost_override: float | None = None) -> dict[str, float]:
    """A run's summary in the dashboard's units (seconds, refusal count), with the real cost when known."""
    s = load_run(filename)["summary"]
    return {
        "evidence_recall": s["evidence_recall"],
        "evidence_mrr": s["evidence_mrr"],
        "context_recall": s.get("context_recall"),
        "context_precision": s.get("context_precision"),
        "faithfulness": s.get("faithfulness"),
        "answer_relevancy": s.get("answer_relevancy"),
        "false_refusals": s["false_refusal_rate"] * n_answerable(),
        "abstention": s.get("abstention"),
        "latency_p50_s": s["latency_p50_ms"] / 1000,
        "latency_p95_s": s["latency_p95_ms"] / 1000,
        "cost_per_query_usd": cost_override if cost_override is not None else s["cost_per_query_usd"],
    }


def scored_sprints() -> list[Sprint]:
    return [s for s in SPRINTS if s.run]


def sprint_by_key(key: str) -> Sprint:
    return next(s for s in SPRINTS if s.key == key)


def questions(filename: str) -> dict[str, dict]:
    """golden_id -> that question's record in a run (output, scores, latency)."""
    return {q["golden_id"]: q for q in load_run(filename)["questions"]}


def evidence_rank(scores: dict) -> int | None:
    """Rank of the evidence chunk among the contexts (MRR = 1/rank), None when it never came back."""
    mrr = scores.get("evidence_mrr") or 0
    return round(1 / mrr) if mrr > 0 else None


# ---------- READMEs ----------

@cache
def readme(folder: str) -> str:
    path = SPRINTS_DIR / folder / "README.md"
    return path.read_text() if path.exists() else ""


def readme_intro(folder: str) -> str:
    """The one-paragraph pitch under the title."""
    lines = readme(folder).splitlines()
    body = [l for l in lines[1:] if l.strip()]
    return body[0] if body else ""


def readme_diagram(folder: str) -> str:
    match = re.search(r"```[a-z]*\n(.*?)```", readme(folder), re.S)
    return match.group(1).rstrip() if match else ""


def readme_section(folder: str, heading: str) -> str:
    """Body of the first '## <heading>' section, up to the next '## '."""
    match = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", readme(folder), re.S | re.M)
    return match.group(1).strip() if match else ""
