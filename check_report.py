"""Phase 7 check: evaluate() runs end to end, saves JSON, and compares two runs.

Run from rag_sprints/:  uv run check_report.py     (free: no LLM calls, free metrics only)
"""

import re
import subprocess
from pathlib import Path

from eval_harness import PipelineOutput, evaluate, load_run, settings

DATA_DIR = Path(__file__).resolve().parent / "data"
PARAGRAPHS = [(p.stem, para) for p in DATA_DIR.glob("*.txt") for para in p.read_text().split("\n\n") if len(para) > 80]


def keyword_pipeline(k: int):
    """Toy pipeline with no LLM: retrieve top-k paragraphs by word overlap, 'answer' with the first one."""
    def pipeline(question: str) -> PipelineOutput:
        words = set(re.findall(r"[a-z0-9]{4,}", question.lower()))
        hits = sorted(PARAGRAPHS, key=lambda dp: -len(words & set(re.findall(r"[a-z0-9]{4,}", dp[1].lower()))))[:k]
        return PipelineOutput(answer=hits[0][1][:200], contexts=[t for _, t in hits], context_ids=[d for d, _ in hits])
    return pipeline


first = evaluate(keyword_pipeline(k=1), arch="toy-k1", use_llm_judge=False, compare_to=None)
first_path = settings.results_dir / f"toy-k1_{first.started_at:%Y%m%d-%H%M%S}.json"

second = evaluate(keyword_pipeline(k=5), arch="toy-k5", use_llm_judge=False, compare_to=first_path)
second_path = settings.results_dir / f"toy-k5_{second.started_at:%Y%m%d-%H%M%S}.json"

# The saved JSON must round-trip back into a RunResult
assert load_run(first_path).summary == first.summary, "saved JSON does not match the run"
print("✅ saved results reload identically")

# The terminal command should print the same comparison
out = subprocess.run(["compare-runs", str(second_path), str(first_path)], capture_output=True, text=True)
print("✅ compare-runs CLI works" if "vs toy-k1" in out.stdout else f"❌ compare-runs failed:\n{out.stderr}")

# Toy runs aren't worth keeping
first_path.unlink()
second_path.unlink()
