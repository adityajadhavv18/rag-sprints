# 07 Self-RAG

Write the answer, then read it back against the chunks. If it's vague, search wider. If it says something the chunks don't, rewrite it.

```
question → Contextual hybrid search (sprint 03) → top 10: ranks 1–5 now, ranks 6–10 saved
         → [IsRel]  sprint 06's grader on ranks 1–5 → keep relevant + partial
         → answer   naive's generate() on the kept chunks
         → check    [IsSup] question + chunks + answer → fully / partially / no + the unsupported claims
                    [IsUse] question + answer only     → useful yes / no            (both calls at once)
         → fully + useful                 → done
         → not useful, or "I don't know"  → wider search ONCE: grade ranks 6–10, answer again, check again
         → useful but not fully supported → rewrite without the unsupported claims  (max 3 rewrites)
         → a brake is hit                 → return the last answer and log a flag
```

Think of an editor reading a finished story next to the reporter's clippings. Sprint 06's fact-checker sorted the clippings *before* the story was written. The editor reads the *story*: "Where does it say that?" and "Would a reader know what to do after this?" A story that fails goes back for a rewrite, or the reporter digs one level deeper.

This follows the Self-RAG paper (Asai et al., 2023). The paper trains a model to emit four reflection tokens: [Retrieve] (search at all?), [IsRel] (is this passage relevant?), [IsSup] (is my answer supported by it?) and [IsUse] (is the answer useful?). Here gpt-4o-mini is *prompted* to play those roles, with no training. [Retrieve] is left out: every question is about this hospital's documents, so the answer is always "search". Medical corpus only.

**Corrective vs Self-RAG in one line:** Corrective checks the chunks before answering. Self-RAG checks the answer after writing it. They catch different mistakes. The chunk check catches a wrong clipping (the HV-RX-2047 look-alike). The answer check catches a bad answer built from okay clippings: a vague one, or one with a fact the model added from its own knowledge.

## How it works

**[IsRel] (`06_corrective_rag/grader.py`, reused).** Sprint 06's grader, unchanged: one call, relevant / partial / irrelevant per chunk, with sprint 03's notes. Sprint 06's refine-and-retry loop is *not* used here; the wider search replaces it.

**[IsSup] (`checker.py`).** One gpt-4o-mini call reads the question, the kept chunks (laid out exactly as the answer step saw them) and the answer. It replies as JSON: first the list of unsupported claims, then the verdict. The list comes first so the model looks before it decides, and the rewrite gets told exactly what to drop.

- **fully:** every fact (number, dose, code, name, step) is stated in the chunks
- **partially:** some are
- **no:** the main facts aren't

A fact stated about a *different* drug, code or device counts as unsupported.

**[IsUse] (`checker.py`).** A second, small call (~150 tokens) that sees **only the question and the answer, no chunks**: "Would this let a nurse act without asking a follow-up?" It runs at the same time as [IsSup], so it adds almost no wait. A garbled reply from either call becomes the cautious value ("partially", "not useful"), so a broken reply leads to a fix, never a silent pass.

**[IsUse] must not see the chunks.** At first [IsUse] was a field inside the [IsSup] call. It called m11's answer "give medicine through the drip" **useful**: the chunks it read were just as vague, so the answer looked complete. On its own, without the chunks, it caught every vague answer:

| [IsUse] test (true answers should be useful, vague ones not) | Inside the [IsSup] call | Stricter wording, same call | **Own call, no chunks** |
|---|---|---|---|
| True answers called useful | 25 / 25 | 25 / 26 | **26 / 26** |
| Vague answers caught | 21 / 25 | 22 / 27 | **27 / 27** |

**The loop (`loop.py`).**

```python
search top 10; grade ranks 1-5; answer
while True:
    if answer isn't "I don't know":
        check → fully + useful? done
    if not useful or "I don't know":
        if not widened: grade ranks 6-10, add keepers, answer again, continue   # once
        if "I don't know": done (honest refusal)
        if fully supported: done, flag "not useful"
    if rewrites == 3: done, flag "unsupported after 3 rewrites"
    rewrite: naive's prompt + the earlier answer + "leave out or correct these claims: …"
```

- **The wider search is ranks 6–10 of the same search**, not the same top 5 again. That was sprint 06's "search wide, not deep" idea. If ranks 6–10 add no kept chunk, there's no new answer call (the answer would be the same).
- **The rewrite says "leave out *or correct*"**, because a changed value (70 mg vs 75 mg) should be fixed, not just deleted.
- **A flagged answer is still returned.** Switching to "I don't know" on a flag would bring back false refusals every time the checker is picky. The flag shows up in the run log.

## Checking the checker

The golden `ground_truth` answers and `evidence` quotes are the answer key, as in sprint 06. There are 4 tests on the 25 answerable questions whose evidence is in the top 10 (m20's isn't). The wrong and vague answers were written once by gpt-4o-mini and saved in `test_answers.json`, so every run uses the same ones.

| Test | Should be | Result |
|---|---|---|
| 1. True answer + evidence chunks | fully, useful | **25 / 25**, useful 25 / 25 |
| 2. One fact changed (dose, code, weight) | not fully | **25 / 25 caught** |
| 3. True answer, evidence chunks removed | not fully | 24 / 25 caught |
| 4. Vague answer | not useful | **25 / 25 caught** |

**A version that failed: cite a chunk per fact.** I tried asking the checker to list every fact with the number of the chunk that states it, and computing the verdict in code. It caught **1 of 25** changed facts. It happily cited the right chunk for a wrong number, matching on topic, not value. A plain "list what's unsupported, then give a verdict" works far better.

**The rewrite works when forced.** No real answer needed a rewrite (see below), so I planted a wrong first answer: m01 "**70** mg" → flagged → rewrite 1 → "**75** mg", fully supported. m24 "under **60** kg" → rewrite 1 → "under **50** kg".

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03 and the notes file. Then, from `rag_sprints/`:

```bash
uv run week2/07_self_rag/checker.py    # the 4 checker tests on 25 golden questions (~$0.04)
uv run week2/07_self_rag/loop.py       # the loop on 10 questions, every step printed (~$0.01)
uv run week2/07_self_rag/run_eval.py   # judged eval vs Corrective + per-question faithfulness + 2nd judge pass (~$0.10)
```

## Results (medical corpus)

| Metric | Contextual hybrid | Corrective | **Self-RAG** |
|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | **0.96 / 0.86** | **0.96 / 0.86** |
| Context recall / precision | 0.92 / 0.76 | 0.95 / **0.90** | **0.96** / **0.90** |
| Answer relevancy | 0.85 | **0.90** | 0.87 |
| Faithfulness on real answers | 0.92 | 0.91 | 0.90 |
| … same answers, judged a 2nd time | – | 0.92 | 0.92 |
| Refused although answerable | 1 / 26 | **0 / 26** | 1 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 |
| Chunks the LLM reads (mean) | 5 | **1.7** | 1.8 |
| Latency p50 / p95 | **1.2 s** / – | 2.6 s / 5.5 s | 2.7 s / **5.0 s** |
| Cost per query | **$0.00031** | $0.00056 | $0.00074 |

**In one line:** Self-RAG fixed m11, the vague answer no earlier sprint could fix, but lost m20 to Corrective. On every other question the answer checker had nothing to remove.

**What the loop did:** 24 questions went straight through (3 calls: grade, answer, check). 6 used the wider search (m11, m20, m27–m30). **0 rewrites, 0 flags.**

### Per question

| Question | Corrective | Self-RAG | Why |
|---|---|---|---|
| m11 "sugar crashed… what should the nurse give?" | "give medicine through the drip" | **"Dextriva 25 g (50 mL) IV push"** | [IsUse] said "not useful". Ranks 6–10 held the evidence at ranks 8 and 9 |
| m20 "potassium through an ordinary arm vein" | **"10 mEq per hour"** | "I don't know" | The evidence is beyond rank 10. Only sprint 06's refined keywords reached it. Self-RAG has no refine step |
| m27–m30 (no answer) | "I don't know" | "I don't know" | Correct, but each spent a wider search for nothing (one extra grader call) |
| the other 24 | ✅ | ✅ same answers | Fully + useful on the first check |

### Why faithfulness didn't move

Faithfulness is Self-RAG's target metric, and it stayed at ~0.91. The answers were already faithful: every fact in the four lowest scorers is in their chunks. The judge (RAGAS, gpt-4o-mini) splits an answer into statements and marks each one, and its own reasons show where it goes wrong:

| Question | Judge | The statement it rejected | What went wrong |
|---|---|---|---|
| m26 alarm light and sound | 0.33 | "The Kestrel IV-220 signals a high-priority alarm with a flashing red light" | The chunk is from the IV-220 manual but starts mid-manual, so the name isn't in it. Sprint 03's old problem, now hitting the judge |
| m12 battery | 0.50 | "The pump is fully charged when it can run for up to 8 hours" | The judge garbled the answer while splitting it, then rejected its own sentence |
| m23 blood culture bottles | 0.60 | "one aerobic bottle" and "one anaerobic bottle", separately | It cut a pair in half, then said each half contradicts "one aerobic AND one anaerobic" |
| m11 Dextriva | 0.75 | "The nurse should administer Dextriva" | "The context doesn't say to give it in every case", though the question describes exactly that case |

The second judge pass gave nearly the same scores (mean change 0.01, 1 of 25 answers moved by 0.2 or more). **The judge isn't random. It makes the same mistakes every time**, so judging twice can't average them out. Our [IsSup] said "fully" on all four, and it was right.

## What it gets wrong

- **It can't find what search didn't bring back (m20).** The wider search only reaches rank 10. Corrective's refined keywords reached further. Self-RAG fixes answers, not queries.
- **The safety net never caught a real fall.** 0 rewrites in 30 questions. gpt-4o-mini, answering from well-graded chunks, almost never adds facts on this corpus. [IsSup] costs ~$0.0002 and ~2,200 prompt tokens per question to confirm what was already true.
- **It checks blind spots with the same model.** Test 3 shows [IsSup] still passed 1 true answer whose evidence was removed. The model that might make a fact up is also the one judging it.
- **Wasted wider searches.** All 4 no-answer questions widened and still (correctly) said "I don't know". An "I don't know" is a cheap reason to widen, but usually a true one.
- **Pricier.** Prompt tokens rose from 3,437 (Corrective) to 4,623 per question, and cost from $0.00056 to $0.00074, which is 2.4× Contextual.

## What it gets right

- **It fixed the vague answer (m11).** Every earlier sprint answered "give medicine through the drip". A chunk check can't see that, because the chunks were on topic. Only a check of the written answer can.
- **Checks run at once, so the wait stays small.** [IsSup] and [IsUse] run in parallel: p50 rose only 0.1 s over Corrective, and p95 is lower (5.0 s vs 5.5 s), because one wider search is cheaper than up to 3 refine rounds.
- **A changed fact is always caught.** 25 / 25 planted wrong values were flagged, and the rewrite fixed them.
- **Abstention still 1.00.** The wider search never turned a no-answer question into a made-up answer.

## Ideas not built

- **Corrective + Self-RAG together.** The two fail in different places: Corrective fixes m20 (refined search), Self-RAG fixes m11 (useful check). Corrective's search loop + this answer check should get both, for ~$0.0008 per query.
- **Skip [IsSup] when the answer copies the chunk.** Most answers repeat their chunk almost word for word. A cheap text-overlap test could skip the call for those and keep it for answers that paraphrase or combine chunks.
- **[Retrieve].** Let the model decide whether to search at all. Not useful on a corpus where every question is about the documents (sprint 05 tested routing).
- **A better judge.** Give the judge the doc labels our answer step gets (fixes m26), or use gpt-4o (fixes the garbled splits). That would change the harness contract, so it's for later.

## When I'd reach for it

When a vague or invented answer is worse than a slow one, and readers act on the exact figures: clinical references, legal or policy Q&A where every statement must trace back to a passage. [IsUse] earns its place when questions are written in plain words over technical text, because that's when answers come out vague. Skip [IsSup] when the model already sticks to its chunks (as here) and every fraction of a cent counts; keep it when the generator is weaker, the context is long, or answers combine many chunks.

## Files

| File | Purpose |
|---|---|
| `checker.py` | [IsSup] and [IsUse] prompts, `check()` (both calls in parallel), and the 4-test check against the golden answers |
| `test_answers.json` | The saved wrong (one fact changed) and vague answers the checker is tested on |
| `loop.py` | `self_rag`: search → [IsRel] → answer → check → widen once / rewrite ×3, and a 10-question demo |
| `pipeline.py` | `self_rag_pipeline`, the harness entry point, plus a log of which path each question took |
| `run_eval.py` | Judged eval on golden v1 (`arch="self_rag"`) vs Corrective, per-question faithfulness, and a 2nd judge pass |
