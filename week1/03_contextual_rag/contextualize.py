"""Contextual RAG, step 1: give every chunk a short "where this comes from" note before it is embedded.

  chunk  = naive's fixed window, cut at 370 tokens instead of 450 so chunk + note fits bge's 512-token limit
  note   = 1-2 sentences from gpt-4o-mini, which reads the WHOLE document plus the chunk
  stored = note + chunk (what the search matches);  the LLM still reads the chunk alone

Notes are cached in notes_medical.json, so they are paid for once. Medical corpus only.

Run from rag_sprints/:
    uv run week1/03_contextual_rag/contextualize.py --sample   # phase 0: notes for 4 chunks + size check
    uv run week1/03_contextual_rag/contextualize.py            # phase 1: notes for every chunk
"""

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

from openai import OpenAI
from tokenizers import Tokenizer

from eval_harness import settings  # importing it loads .env, so OpenAI() finds the key
from eval_harness.golden import load_golden
from eval_harness.metrics import _evidence_fragments, _norm

os.environ.setdefault("RAG_CORPUS", "medical")
# Both sprints have a file called ingest.py, so load naive's by path under another name
_spec = importlib.util.spec_from_file_location(
    "naive_ingest", Path(__file__).resolve().parents[1] / "00_naive_baseline" / "ingest.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)
if naive.CORPUS != "medical":
    raise SystemExit("Contextual RAG runs on the medical corpus only: unset RAG_CORPUS or set it to medical")

GOLDEN_SET, EMBED_MODEL = naive.GOLDEN_SET, naive.EMBED_MODEL
CHUNK_TOKENS = 370  # 370 chunk + up to ~130 note + 2 special tokens stays under bge's 512
BGE_LIMIT = 512
NOTES_FILE = Path(__file__).with_name("notes_medical.json")
# gpt-4o-mini, $ per million tokens. Repeated prompt starts (the document) are cached by OpenAI at half price
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.15, 0.075, 0.60

# Anthropic's contextual-retrieval prompt, plus two rules for this corpus. The document goes FIRST, so every
# chunk of the same document shares the same opening text and OpenAI can reuse it from its cache.
PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Write a short context (at most 2 sentences, under 50 words) that situates this chunk within the document, \
to improve search retrieval of the chunk.
- Name the document, and every entry the chunk covers by its code and name (for example "HV-RX-1234 Drugname"), \
including an entry the chunk starts or ends in the middle of.
- Say what kind of information it holds (for example dosing, fault codes, alarm actions), adding everyday words \
where they help (for example "heart" for "cardiac").
Answer only with the context and nothing else."""


def load_chunks() -> list:
    """Naive's chunker, unchanged except for the window size."""
    naive.CHUNK_TOKENS = CHUNK_TOKENS  # chunk_document reads this module setting each time it runs
    return naive.load_chunks()


def contextualized_text(note: str, chunk_text: str) -> str:
    """What gets embedded and indexed: the note on top, then the original chunk."""
    return f"{note}\n\n{chunk_text}"


def load_notes() -> dict:
    return json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else {}


def write_note(client: OpenAI, document: str, chunk) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        max_tokens=100,  # hard stop in case the model ignores "under 50 words"
        messages=[{"role": "user", "content": PROMPT.format(document=document, chunk=chunk.text)}],
    )
    u = resp.usage
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0
    return resp.choices[0].message.content.strip(), {"in": u.prompt_tokens, "cached": cached, "out": u.completion_tokens}


def add_notes(chunks: list) -> dict:
    """Write a note for every chunk that has none in the cache yet; save after each one, so a crash loses nothing."""
    notes = load_notes()
    docs = {p.stem: p.read_text() for p in naive.DATA_DIR.glob("*.txt")}
    client = OpenAI()
    usage = {"in": 0, "cached": 0, "out": 0, "calls": 0}
    for chunk in chunks:
        saved = notes.get(chunk.chunk_id)
        # Same id but different text means the chunking changed: the old note describes another chunk
        if saved and saved["chunk_start"] == chunk.text[:80]:
            continue
        note, u = write_note(client, docs[chunk.doc_id], chunk)
        notes[chunk.chunk_id] = {"note": note, "chunk_start": chunk.text[:80]}
        NOTES_FILE.write_text(json.dumps(notes, indent=1, ensure_ascii=False))
        for key in ("in", "cached", "out"):
            usage[key] += u[key]
        usage["calls"] += 1
    return usage


def cost(usage: dict) -> float:
    fresh = usage["in"] - usage["cached"]
    return (fresh * PRICE_IN + usage["cached"] * PRICE_CACHED + usage["out"] * PRICE_OUT) / 1e6


def sample_chunks(chunks: list) -> list:
    """The chunk holding the evidence for questions we expect notes to help (m01, m07, m09) or maybe not (m20)."""
    golden = {g.id: g for g in load_golden(GOLDEN_SET)}
    picks = []
    for qid in ("m01", "m07", "m09", "m20"):
        fragments = _evidence_fragments(golden[qid])
        picks.append((qid, next(c for c in chunks if all(f in _norm(c.text) for f in fragments))))
    return picks


def main():
    tokenizer = Tokenizer.from_pretrained(EMBED_MODEL)
    tokenizer.no_truncation()

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(text).ids)  # includes bge's 2 special tokens, as the embedder sees it

    chunks = load_chunks()
    sizes = [c.n_tokens for c in chunks]
    print(f"1. chunked: {len(chunks)} chunks of up to {CHUNK_TOKENS} tokens (naive at 450: 147 chunks)")
    print(f"     chunk tokens median {statistics.median(sizes):.0f}, max {max(sizes)}  "
          f"-> room left for a note: {BGE_LIMIT - 2 - max(sizes)} tokens")

    sample = "--sample" in sys.argv
    picks = sample_chunks(chunks)
    usage = add_notes([c for _, c in picks] if sample else chunks)
    notes = load_notes()
    print(f"2. notes: {usage['calls']} new ({usage['in']:,} input tokens, {usage['cached']:,} of them cached, "
          f"{usage['out']:,} output) cost ${cost(usage):.4f}.  {len(notes)} in {NOTES_FILE.name}")

    print("3. examples")
    for qid, c in picks:
        note = notes[c.chunk_id]["note"]
        total = n_tokens(contextualized_text(note, c.text))
        print(f"\n  [{qid}] {c.chunk_id}   chunk {c.n_tokens} + note {n_tokens(note) - 2} = {total} tokens "
              f"{'OK' if total <= BGE_LIMIT else 'OVER 512'}")
        print(f"    NOTE:  {note}")
        print(f"    CHUNK: {c.text[:160].replace(chr(10), ' | ')} …")

    done = [c for c in chunks if c.chunk_id in notes]
    over = [c for c in done if n_tokens(contextualized_text(notes[c.chunk_id]["note"], c.text)) > BGE_LIMIT]
    print(f"\n4. size check: {len(over)} of {len(done)} noted chunks go over {BGE_LIMIT} tokens")


if __name__ == "__main__":
    main()
