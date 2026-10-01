"""Phase 2 check: valid data passes, broken data is rejected.

Run from rag_sprints/:  uv run check_schema.py
"""

from pydantic import ValidationError

from eval_harness import GoldenItem, PipelineOutput


def expect_fail(label, fn):
    try:
        fn()
        print(f"❌ {label}: should have been rejected")
    except ValidationError as e:
        print(f"✅ {label}: rejected -> {e.errors()[0]['msg']}")


# Valid data
GoldenItem(id="q01", question="What was Apple's FY2024 revenue?", ground_truth="$391B", source_doc="aapl_10k_2024")
GoldenItem(id="q30", question="What is Apple's CEO's favourite colour?", ground_truth="Not in the documents.",
           question_type="no_answer")
PipelineOutput(answer="$391B", contexts=["Total net sales were $391B"], context_ids=["aapl_10k_2024"], tokens_in=120)
print("✅ valid GoldenItem + PipelineOutput accepted")

# Broken data
expect_fail("answerable question with no source_doc",
            lambda: GoldenItem(id="q02", question="What was revenue?", ground_truth="x"))
expect_fail("no_answer question with a source_doc",
            lambda: GoldenItem(id="q03", question="Unanswerable?", ground_truth="x",
                               source_doc="aapl", question_type="no_answer"))
expect_fail("unknown question_type",
            lambda: GoldenItem(id="q04", question="What was revenue?", ground_truth="x",
                               source_doc="aapl", question_type="hard"))
expect_fail("contexts / context_ids length mismatch",
            lambda: PipelineOutput(answer="x", contexts=["a", "b"], context_ids=["d1"]))
expect_fail("typo in field name",
            lambda: PipelineOutput(answer="x", contexs=["a"], contexts=["a"], context_ids=["d1"]))
expect_fail("negative token count",
            lambda: PipelineOutput(answer="x", contexts=[], context_ids=[], tokens_in=-5))
