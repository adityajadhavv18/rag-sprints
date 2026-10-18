"""GraphRAG lite, retrieval: question -> entities it names -> a guided walk on the graph -> the walked links' chunks.

  1. find   string-match node names in the question ("Corvetrol", "septic shock", "LC-0625"). Free.
  2. walk   up to 3 hops. At each hop, score every arrow leaving the NEW nodes we reached (arrow + the sentence it
            came from) against the question
            (local bge embeddings, free) and keep only the best 3. A hiker reading the signposts at every fork,
            not walking every trail: Vancorin alone has 43 arrows, and a blind 2-hop walk would bring half the formulary.
  3. collect the chunks those arrows came from, in walk order (hop 1 first)

Run from rag_sprints/:  uv run week2/09_graph_rag/retrieve.py
Walks the 5 chain questions, prints every hop, and checks each hop's evidence: graph walk vs Contextual search alone.
Then a quick count over the 30 golden questions: how many name an entity the graph knows. Free (no LLM calls).
Needs graph_medical.json (build_graph.py) and Qdrant running for the search comparison.
"""

import json
import re
from dataclasses import dataclass, field
from functools import cache

import numpy as np
from fastembed import TextEmbedding

from build_graph import CODE, ROOT, _load, code_titles, contextual, load_graph, name_to_code
from eval_harness.golden import load_golden
from eval_harness.metrics import _evidence_fragments, _norm

HOPS = 3          # c05 needs 3: septic shock -> Norvessa -> Central pharmacy -> extension 4200
BEAM = 3          # places reached per hop
MIN_NAME = 4      # shorter names ("IV", "INR") match too much by accident; codes always count
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # the same local model as every sprint's dense search

# How an arrow reads to the embedder: "HV-RX-6080 Norvessa: call Central pharmacy"
PHRASES = {"monitored_by": "is monitored with the lab test", "reversed_by": "is reversed by",
           "interacts_with": "interacts with", "incompatible_with": "is incompatible on the same IV line with",
           "compatible_with": "is compatible on the same IV line with", "treats": "treats",
           "first_line_for": "is the first-line drug for", "uses_drug": "gives the drug",
           "orders_test": "orders the lab test", "covers": "is the protocol for", "susceptible_to": "is treated by",
           "has_fault": "can show fault code", "contact": ": call", "has_extension": "phone extension is",
           "related_to": "is related to", "mentions": "mentions"}


@dataclass
class Step:
    hop: int
    head: str
    relation: str
    tail: str
    score: float
    chunks: list[str]


@dataclass
class Walk:
    seeds: list[str]
    steps: list[Step] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)  # walk order, no repeats


@cache
def graph():
    return load_graph()


@cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


def _vectors(texts: list[str]) -> np.ndarray:
    return np.array(list(_embedder().embed(texts)))  # bge vectors come normalised: dot product = cosine


def edge_text(u: str, rel: str, v: str) -> str:
    g = graph()
    return f"{g.nodes[u]['name']} {PHRASES.get(rel, rel)} {g.nodes[v]['name']}"


def signpost(u: str, rel: str, v: str) -> str:
    """What the walk scores: the arrow plus the sentence that states it. The sentence carries the condition
    ("If extravasation occurs ... call pharmacy") that the bare arrow loses."""
    sentences = graph()[u][v][rel].get("sentences", [])
    return edge_text(u, rel, v) + (f". {sentences[0]}" if sentences else "")


@cache
def edge_vectors() -> dict[tuple[str, str, str], np.ndarray]:
    """Every arrow embedded once (~1,400 short texts, a few seconds on the laptop)."""
    edges = list(graph().edges(keys=True))
    return dict(zip(edges, _vectors([signpost(u, k, v) for u, v, k in edges])))


# ---------- 1. Find: which nodes does the question name? ----------

@cache
def name_index() -> dict[str, str]:
    """lower-case name -> node key. Node names, plus every name an entry answers to ("aptt" -> LC-0625)."""
    g, index = graph(), {}
    for key, data in g.nodes(data=True):
        name = data["name"].lower()
        for form in {data["name"], data["name"].split(" (")[0], key}:
            if " " not in form and form.islower():
                continue  # "monitor", "staff", "level": one lower-case word is an ordinary word, not a name
                          # (types can't tell: the LLM typed "level" as a protocol). Names look like Corvetrol, LC-0625
            if CODE.fullmatch(form) or len(form) >= MIN_NAME:
                index.setdefault(form.lower(), key)
        if CODE.fullmatch(key):  # "HV-RX-2074 Corvetrol" also answers to "corvetrol"
            index.setdefault(name.removeprefix(key.lower()).strip(), key)
    for n, code in name_to_code().items():
        if code in g and len(n) >= MIN_NAME:
            index.setdefault(n, code)
    return index


def find_entities(question: str) -> list[str]:
    """Whole-word matches, longest first; a match inside a longer one is dropped ("ketoacidosis" in
    "diabetic ketoacidosis")."""
    q = question.lower()
    taken, found = [], []
    for name in sorted(name_index(), key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", q):
            if any(s <= m.start() and m.end() <= e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            key = name_index()[name]
            if key not in found:
                found.append(key)
    return found


# ---------- 2. Walk: follow the arrows that best match the question ----------

def walk(question: str, hops: int = HOPS, beam: int = BEAM) -> Walk:
    g, vecs = graph(), edge_vectors()
    q = _vectors([question])[0]
    result = Walk(seeds=find_entities(question))
    frontier, used, visited, walked = set(result.seeds), set(), set(result.seeds), set()
    for hop in range(1, hops + 1):
        # every arrow touching the frontier, in either direction (a link is a link, whichever way it was written)
        options = {(u, v, k) for n in frontier for u, v, k in [*g.out_edges(n, keys=True), *g.in_edges(n, keys=True)]}
        options -= used
        if not options:
            break
        best, places = [], set()
        for u, v, k in sorted(options, key=lambda e: -float(vecs[e] @ q)):
            # the beam picks PLACES, not arrows: two arrows into CP-07 take one slot. An arrow between two places
            # we already know (Vancorin <-> Norvessa in c04) is a place of its own: that link may be the answer
            new = [n for n in (u, v) if n not in visited]
            place = new[0] if new else frozenset((u, v))
            if place not in places and frozenset((u, v)) not in walked:  # same pair, another label: skip
                places.add(place)
                best.append((u, v, k))
            if len(best) == beam:
                break
        frontier = set()
        for u, v, k in best:
            used.add((u, v, k))
            walked.add(frozenset((u, v)))
            chunks = g[u][v][k]["chunks"]
            result.steps.append(Step(hop, u, k, v, float(vecs[(u, v, k)] @ q), chunks))
            result.chunk_ids += [c for c in chunks if c not in result.chunk_ids]
            frontier |= {u, v} - visited  # only NEW nodes: never walk back out of where we've been
        visited |= frontier
    return result


# ---------- Demo: the chain questions, hop by hop ----------

@cache
def chunk_text() -> dict[str, str]:
    return {c.chunk_id: c.text for c in contextual().load_chunks()}


def has(evidence: str, chunk_ids: list[str]) -> str:
    """Which position (1-based) holds this evidence quote, or '–' if none."""
    for i, cid in enumerate(chunk_ids, 1):
        if _norm(evidence) in _norm(chunk_text()[cid]):
            return f"#{i}"
    return "–"


def main():
    search = _load("contextual_pipeline", ROOT / "week1" / "03_contextual_rag" / "pipeline.py").hybrid_search
    g = graph()
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} arrows; walk = {HOPS} hops x best {BEAM}")
    chains = [json.loads(line) for line in (ROOT / "week1" / "05_adaptive_rag" / "questions_extra.jsonl").open()]
    totals = {"walk": 0, "search": 0, "both": 0, "hops": 0}
    for item in (c for c in chains if c["id"].startswith("c")):
        w = walk(item["question"])
        hits = [h.chunk_id for h in search(item["question"])]
        print(f"\n{'=' * 100}\n{item['id']}: {item['question']}")
        print(f"  entities found: {[g.nodes[s]['name'] for s in w.seeds] or 'none'}")
        for s in w.steps:
            print(f"  hop {s.hop}  {s.score:.2f}  {edge_text(s.head, s.relation, s.tail)}   [{', '.join(s.chunks)}]")
        print(f"  chunks from the walk: {len(w.chunk_ids)}")
        print(f"  {'hop evidence':58} {'walk':>6} {'search':>7} {'both':>5}")
        for hop in item["hops"]:
            a, b = has(hop["evidence"], w.chunk_ids), has(hop["evidence"], hits)
            both = has(hop["evidence"], hits + w.chunk_ids)
            totals["walk"] += a != "–"; totals["search"] += b != "–"; totals["both"] += both != "–"; totals["hops"] += 1
            print(f"  {hop['doc'].removeprefix('hvmc_') + ': ' + hop['evidence'][:40]:58} {a:>6} {b:>7} {both:>5}")
    n = totals["hops"]
    print(f"\n{'=' * 100}\nhop evidence found, all 5 chains:  walk {totals['walk']}/{n}   search top 5 "
          f"{totals['search']}/{n}   search + walk {totals['both']}/{n}")

    # How often does the graph get a say on the golden set?
    golden = load_golden(ROOT.parent / "eval_harness" / "golden_sets" / "golden_medical_v1.jsonl")
    print(f"\nGolden set ({len(golden)} questions): entities found, chunks walked, evidence in the walk")
    rows, named, walked_ev = [], 0, 0
    for item in golden:
        w = walk(item.question)
        frags = _evidence_fragments(item)
        ev = bool(frags) and all(any(f in _norm(chunk_text()[c]) for c in w.chunk_ids) for f in frags)
        named += bool(w.seeds); walked_ev += ev
        rows.append(f"  {item.id} {item.question_type:11} {len(w.seeds)} ent  {len(w.chunk_ids)} chunks  "
                    f"{'evidence ✅' if ev else '':11} {', '.join(g.nodes[s]['name'] for s in w.seeds)[:60]}")
    print("\n".join(rows))
    print(f"\n{named}/{len(golden)} name at least one entity; the walk alone holds all the evidence for {walked_ev}")


if __name__ == "__main__":
    main()
