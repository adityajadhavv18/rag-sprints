"""Phase 4 check: the runner times every question and survives crashing / contract-breaking pipelines.

Run from rag_sprints/:  uv run check_runner.py
"""

from openai import OpenAI

from eval_harness import PipelineOutput, load_golden, run_pipeline, settings

golden = load_golden()


# Test 1: a fake pipeline that misbehaves on purpose
def dummy_pipeline(question: str) -> PipelineOutput:
    if "shareholders of record" in question:  # q04
        raise RuntimeError("simulated vector DB timeout")
    if "Microsoft's commercial remaining" in question:  # q12
        return {"answer": "x", "contexts": ["a", "b"], "context_ids": ["d1"]}  # misaligned -> contract violation
    return PipelineOutput(answer="I don't know.", contexts=["fake chunk"], context_ids=["aapl_10k_2025"])


print("Test 1: dummy pipeline over all 30 questions")
results = run_pipeline(dummy_pipeline, golden, verbose=False)
errors = [r for r in results if r.error]
print(f"  ran {len(results)}/30, {len(errors)} errors (expected 2):")
for r in errors:
    print(f"    {r.golden_id}: {r.error[:90]}")


# Test 2: a real LLM with NO retrieval, on 3 questions; proves timing + token capture on real calls
client = OpenAI()


def llm_only_pipeline(question: str) -> PipelineOutput:
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": f"Answer in one sentence: {question}"}],
    )
    return PipelineOutput(
        answer=resp.choices[0].message.content,
        contexts=[],
        context_ids=[],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )


print("\nTest 2: real LLM (no retrieval) on 3 questions")
for r in run_pipeline(llm_only_pipeline, golden[:3]):
    print(f"    {r.golden_id} tokens {r.output.tokens_in}/{r.output.tokens_out}: {r.output.answer[:110]}")
