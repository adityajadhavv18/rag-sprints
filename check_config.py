"""Phase 1 check: confirms .env is loaded and required keys are set.

Run from rag_sprints/:  uv run check_config.py
"""

from eval_harness import settings

print(f"llm_model:   {settings.llm_model}")
print(f"judge_model: {settings.judge_model}")
print(f"embed_model: {settings.embed_model}")
print(f"tracing:     {settings.langsmith_tracing} (project: {settings.langsmith_project})")
print(f"results_dir: {settings.results_dir}")

missing = settings.missing_keys()
if missing:
    print(f"\n❌ Missing keys in .env: {', '.join(missing)}")
else:
    print("\n✅ All keys set — Phase 1 passed")
