"""GraphRAG lite, indexing: every chunk -> (entity, relation, entity) triples -> one graph -> graph_medical.json.

  extract  one gpt-4o-mini call per chunk (sprint 03's 180 chunks, each with its note), relations from a fixed list
           + code links, free: an entry line that names another code ("Monitoring: aPTT (LC-0625)") is a link.
           Codes are regular, so code reads them; the LLM reads the prose links ("start Norvessa, the first-line
           vasopressor"). Two LLM runs disagreed on whether Corvetrol -> LC-0625 exists; the regex never does
  merge    in code: "Corvetrol", "Corvetrol (HV-RX-2074)" and "HV-RX-2074" become ONE node (keyed by the code);
           fault codes get their device in front, so the IV220's E-03 never merges with the SP40's E-03
  store    a networkx graph saved as JSON. Every edge remembers the chunk(s) it came from, so a walk on the graph
           can hand real chunk text to the LLM later, and the sentence that states it (its "why": the bare triple
           "Norvessa -> contact -> pharmacy" loses "if it leaks out of the vein")

Think of the "see also" notes in a book's margins: the chunks are the pages, the graph is the cross-references.

Triples are cached in triples_medical.json, so extraction is paid for once. Medical corpus only.

Run from rag_sprints/:
    uv run week2/09_graph_rag/build_graph.py --sample   # extract the 6 chunks the chain questions need, print them
    uv run week2/09_graph_rag/build_graph.py            # extract every chunk, build and save the graph, draw it
"""

import importlib.util
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import networkx as nx
from openai import OpenAI

from eval_harness import settings  # importing it loads .env, so OpenAI() finds the key

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # rag_sprints/
TRIPLES_FILE = HERE / "triples_medical.json"
GRAPH_FILE = HERE / "graph_medical.json"
COST_FILE = HERE / "index_cost.json"
PICTURE_FILE = HERE / "chain_neighbourhood.png"
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens
WORKERS = 8

ENTITY_TYPES = ("drug", "lab_test", "protocol", "condition", "device", "fault_code", "department", "extension",
                "organism", "other")
# (relation, head type -> tail type). A short fixed list keeps the graph readable; related_to catches the rest.
RELATIONS = {
    "monitored_by": "drug -> lab_test that monitors it",
    "reversed_by": "drug -> drug that reverses it",
    "interacts_with": "drug -> drug it interacts with",
    "incompatible_with": "drug -> drug it must not share an IV line / Y-site with",
    "compatible_with": "drug -> drug it may share a Y-site with",
    "treats": "drug -> condition",
    "first_line_for": "drug -> condition or protocol it is the first choice for",
    "uses_drug": "protocol -> drug it gives",
    "orders_test": "protocol -> lab_test it orders",
    "covers": "protocol -> condition it is about",
    "susceptible_to": "organism -> drug that works on it",
    "has_fault": "device -> fault_code it can show",
    "contact": "anything -> department to call or page about it",
    "has_extension": "department -> extension (phone number)",
    "related_to": "any other link worth following",
}
CODE_ONLY = {"mentions": "entry -> a code its text names (made by code_links, never by the LLM)"}
# Relations whose direction the types decide. A triple that arrives as (lab_test orders_test protocol) gets flipped.
TYPED = {("monitored_by", "drug", "lab_test"), ("uses_drug", "protocol", "drug"), ("orders_test", "protocol", "lab_test"),
         ("covers", "protocol", "condition"), ("treats", "drug", "condition"), ("has_fault", "device", "fault_code"),
         ("has_extension", "department", "extension"), ("susceptible_to", "organism", "drug")}

PROMPT = """You build a knowledge graph from one hospital's internal documents.
Below is one chunk, with a note saying which document and entries it comes from. Chunks often start or end in \
the middle of an entry: use the note to tell which entry a sentence belongs to.

List the links between named things in this chunk as JSON:
{{"triples": [{{"head": "...", "head_type": "...", "relation": "...", "tail": "...", "tail_type": "...", \
"sentence": "..."}}]}}
"sentence" is the shortest sentence or clause from the chunk that states the link, copied word for word \
(under 30 words). Keep the condition it depends on, e.g. "If extravasation occurs, stop the infusion ... and \
call pharmacy for the antidote".

Entity types: {types}
Relations (head -> tail):
{relations}

Work through the chunk entry by entry and line by line: every line that names another drug, test, protocol, \
condition, department or code is a link. Typical lines:
- "Monitoring: aPTT (LC-0625)" in a drug entry -> drug monitored_by "LC-0625 aPTT"
- "The reversal agent is X" -> drug reversed_by X
- "call pharmacy", "page the rapid response team", "notify the blood bank" -> the entry's subject contact that department
- "start X, which is the first-line vasopressor" -> X first_line_for the condition or protocol

Rules:
- Name entities exactly as the document does. Add the code when the chunk gives one, e.g. "HV-RX-2074 Corvetrol", \
"LC-0625 aPTT", "CP-07 Sepsis recognition".
- Fault and alarm codes: always put the device first, e.g. "Kestrel IV220 E-03".
- Phone extensions are entities: "extension 4200". Link them to their department with has_extension.
- Only link named things. Doses, times, ranges and numbers stay in the text: never make them entities.
- Skip generic words (patient, nurse, physician, IV, blood, tube, hospital).
- Only links the chunk actually states. An empty list is fine.

<note>{note}</note>
<chunk>{chunk}</chunk>"""


# ---------- Loading sprint 03's chunks and sprint 08's entry table ----------

def _load(name: str, path: Path):
    """Load another sprint's file by path (folders start with digits, so a normal import can't reach them)."""
    sys.path.append(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def contextual():
    """Sprint 03's contextualize.py: load_chunks() and load_notes()."""
    return _load("contextualize", ROOT / "week1" / "03_contextual_rag" / "contextualize.py")


@cache
def name_to_code() -> dict[str, str]:
    """Every name an entry answers to -> its code, from sprint 08's lookup table: "corvetrol" -> HV-RX-2074,
    "aptt" -> LC-0625. A name shared by two entries (e.g. "potassium") is left out: we can't tell which one."""
    tools = _load("agentic_tools", ROOT / "week2" / "08_agentic_rag" / "tools.py")
    main, alias = {}, {}  # main name = the title before any bracket; alias = a one-word bracket ("aPTT")
    for code, (_, block, names) in tools._entries().items():
        title = block.split("\n")[0][len(code) + 1:].lower()
        for n in names:
            bucket = main if n in (title, title.split(" (")[0], title.split(" (")[0].split(",")[0]) else alias
            bucket.setdefault(n, set()).add(code)
    # A main name beats another entry's alias: "Heparivex" is HV-RX-2010, not LC-0920 "Anti-Xa level (Heparivex)"
    merged = {n: codes for n, codes in alias.items() if n not in main} | main
    return {n: codes.pop() for n, codes in merged.items() if len(codes) == 1}


@cache
def code_titles() -> dict[str, str]:
    """code -> "HV-RX-2074 Corvetrol", the display name for code nodes."""
    tools = _load("agentic_tools", ROOT / "week2" / "08_agentic_rag" / "tools.py")
    return {code: f"{code} {block.split(chr(10))[0][len(code) + 1:].split(' (')[0]}"
            for code, (_, block, _) in tools._entries().items()}


# ---------- Merging names into one node key (code, not prompt: a rule in Python always holds) ----------

CODE = re.compile(r"\b(HV-RX-\d{4}|LC-\d{4}|CP-\d{2})\b", re.IGNORECASE)
FAULT = re.compile(r"\b([A-Z]{1,2}-\d{2,3})\b")
DEVICES = {"kestrel_iv220_manual": "Kestrel IV220", "kestrel_pca90_manual": "Kestrel PCA90",
           "kestrel_sp40_manual": "Kestrel SP40", "talon_vm8_monitor_manual": "Talon VM8"}


def node_key(name: str, etype: str, doc_id: str) -> tuple[str, str]:
    """(key, display name). The key decides which mentions become the same node."""
    name = " ".join(name.split()).strip(" .,;:")
    code = CODE.search(name)
    if code:
        key = code.group(1).upper()
        return key, code_titles().get(key, name)
    low = name.lower().removeprefix("the ")
    if low in name_to_code():
        key = name_to_code()[low]
        return key, code_titles().get(key, name)
    ext = re.fullmatch(r"(?:ext\.?|extension)\s*(\d{4})", low)
    if ext:
        return f"ext {ext.group(1)}", f"extension {ext.group(1)}"
    if etype == "fault_code" or FAULT.fullmatch(name):
        device = DEVICES.get(doc_id)
        fault = FAULT.search(name)
        if device and fault and not low.startswith(device.lower()):
            name = f"{device} {fault.group(1)}"  # the LLM forgot the device: add it from the chunk's document
        return name.lower(), name
    return low, name


# ---------- Extraction ----------

@cache
def _openai() -> OpenAI:
    return OpenAI()


def extract(chunk, note: str) -> dict:
    """One LLM call -> the chunk's triples plus token counts. Unknown relations become related_to."""
    prompt = PROMPT.format(types=", ".join(ENTITY_TYPES), note=note, chunk=chunk.text,
                           relations="\n".join(f"- {r}: {d}" for r, d in RELATIONS.items()))
    for attempt in range(3):
        try:
            resp = _openai().chat.completions.create(model="gpt-4o-mini", temperature=0,
                                                     response_format={"type": "json_object"},
                                                     messages=[{"role": "user", "content": prompt}])
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    try:
        raw = json.loads(resp.choices[0].message.content).get("triples", [])
    except json.JSONDecodeError:
        raw = []
    triples = []
    for t in raw:
        if not all(isinstance(t.get(f), str) and t[f].strip() for f in ("head", "relation", "tail")):
            continue
        t = {"head": t["head"].strip(), "head_type": t.get("head_type", "other"),
             "relation": t["relation"] if t["relation"] in RELATIONS else "related_to",
             "tail": t["tail"].strip(), "tail_type": t.get("tail_type", "other"),
             "sentence": str(t.get("sentence", "")).strip()}
        if (t["relation"], t["tail_type"], t["head_type"]) in TYPED:  # written backwards: flip it
            t = {"head": t["tail"], "head_type": t["tail_type"], "relation": t["relation"],
                 "tail": t["head"], "tail_type": t["head_type"], "sentence": t["sentence"]}
        triples.append(t)
    return {"chunk_start": chunk.text[:80], "triples": triples,
            "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens}


def extract_all(chunks, notes) -> dict:
    """Extract every chunk not already in the cache (or whose text changed), in parallel. Saves as it goes."""
    cache_ = json.loads(TRIPLES_FILE.read_text()) if TRIPLES_FILE.exists() else {}
    todo = [c for c in chunks if cache_.get(c.chunk_id, {}).get("chunk_start") != c.text[:80]]
    if todo:
        print(f"extracting {len(todo)} chunk(s) with {WORKERS} workers ...")
        with ThreadPoolExecutor(WORKERS) as pool:
            for c, result in zip(todo, pool.map(lambda c: extract(c, notes[c.chunk_id]["note"]), todo)):
                cache_[c.chunk_id] = result
        TRIPLES_FILE.write_text(json.dumps(cache_, indent=1))
    return cache_


# ---------- The graph ----------

CODE_TYPES = {"HV-RX": "drug", "LC": "lab_test", "CP": "protocol"}


def code_links(chunks, notes) -> dict[str, list[dict]]:
    """chunk_id -> triples found by regex: inside an entry, every line naming ANOTHER entry's code is a link.

    The link is credited to the chunk that holds the line and belongs to that entry (its text or its sprint 03
    note names the entry's code), so the walk can hand that chunk to the LLM.
    """
    tools = _load("agentic_tools", ROOT / "week2" / "08_agentic_rag" / "tools.py")
    found: dict[str, list[dict]] = {}
    for code, (doc_id, block, _) in tools._entries().items():
        for line in block.split("\n")[1:]:
            for target in {m.upper() for m in CODE.findall(line)} - {code}:
                rel = "monitored_by" if line.startswith("Monitoring:") and target.startswith("LC") else "mentions"
                where = [c for c in chunks if c.doc_id == doc_id and line[:60] in " ".join(c.text.split())
                         and (code in c.text or code in notes[c.chunk_id]["note"])]
                if not where:  # the line is cut by a chunk boundary: fall back to the chunk naming the target
                    where = [c for c in chunks if c.doc_id == doc_id and target in c.text
                             and (code in c.text or code in notes[c.chunk_id]["note"])]
                if where:
                    found.setdefault(where[0].chunk_id, []).append(
                        {"head": code, "head_type": CODE_TYPES[code.rsplit("-", 1)[0]], "relation": rel,
                         "tail": target, "tail_type": CODE_TYPES[target.rsplit("-", 1)[0]], "sentence": line[:200]})
    return found


def department_aliases(chunks, triples: dict) -> dict[str, str]:
    """A department name that is the end of exactly ONE longer department name merges into it:
    "pharmacy" -> "central pharmacy" (the formulary says "call pharmacy", the directory "Central pharmacy")."""
    depts = {node_key(t[side], "department", c.doc_id)[0] for c in chunks for t in triples[c.chunk_id]["triples"]
             for side in ("head", "tail") if t[f"{side}_type"] == "department"}
    alias = {}
    for short in depts:
        longer = [d for d in depts if d.endswith(" " + short)]
        if len(longer) == 1:
            alias[short] = longer[0]
    return alias


def build(chunks, triples: dict) -> nx.MultiDiGraph:
    """One node per key, one edge per (head, relation, tail). Same link stated in 3 chunks = 1 edge, 3 chunk ids."""
    g = nx.MultiDiGraph()
    doc_of = {c.chunk_id: c.doc_id for c in chunks}
    alias = department_aliases(chunks, triples)
    for c in chunks:
        for t in triples[c.chunk_id]["triples"]:
            ends = []
            for side in ("head", "tail"):
                key, name = node_key(t[side], t[f"{side}_type"], c.doc_id)
                key = alias.get(key, key)
                if key not in g:
                    g.add_node(key, name=name, type=t[f"{side}_type"], chunks=[])
                if c.chunk_id not in g.nodes[key]["chunks"]:
                    g.nodes[key]["chunks"].append(c.chunk_id)
                ends.append(key)
            head, tail = ends
            if head == tail:
                continue
            rel = t["relation"]
            if g.has_edge(head, tail, key=rel):
                if c.chunk_id not in g[head][tail][rel]["chunks"]:
                    g[head][tail][rel]["chunks"].append(c.chunk_id)
            else:
                g.add_edge(head, tail, key=rel, relation=rel, chunks=[c.chunk_id], docs=[], sentences=[])
            sentences = g[head][tail][rel]["sentences"]
            if t.get("sentence") and t["sentence"] not in sentences:
                sentences.append(t["sentence"])  # the "why" of the link (GraphRAG's relationship description)
            docs = g[head][tail][rel]["docs"]
            if doc_of[c.chunk_id] not in docs:
                docs.append(doc_of[c.chunk_id])
    return g


def save(g: nx.MultiDiGraph):
    GRAPH_FILE.write_text(json.dumps(nx.node_link_data(g, edges="edges"), indent=1))


def load_graph() -> nx.MultiDiGraph:
    return nx.node_link_graph(json.loads(GRAPH_FILE.read_text()), edges="edges", multigraph=True, directed=True)


# ---------- Reports ----------

# The chain questions' entities (questions_extra.jsonl c01-c05): what we'll look at in the picture.
CHAIN_SEEDS = ["HV-RX-2074", "LC-0625", "LC-0870", "CP-27", "LC-0452", "CP-07", "ext 4200"]
CHAIN_NAMES = ["thyrolex", "norvessa", "vancorin", "central pharmacy", "septic shock"]


def describe(g: nx.MultiDiGraph, key: str) -> str:
    return f"{g.nodes[key]['name']}" if key in g else f"{key} (not in graph)"


def edge_line(g, u, v, rel) -> str:
    return f"  {describe(g, u)}  --{rel}-->  {describe(g, v)}   [{', '.join(g[u][v][rel]['chunks'])}]"


def chain_check(g: nx.MultiDiGraph):
    """Is each chain hop a link in the graph? (the walk in stage 2 can only follow links that exist)"""
    hops = [("c01", "HV-RX-2074", "LC-0625"), ("c02", node_key("Thyrolex", "drug", "")[0], "LC-0870"),
            ("c03", "CP-27", "LC-0452"), ("c04", "CP-07", node_key("Norvessa", "drug", "")[0]),
            ("c04", node_key("Norvessa", "drug", "")[0], node_key("Vancorin", "drug", "")[0]),
            ("c05", node_key("Norvessa", "drug", "")[0], "central pharmacy"), ("c05", "central pharmacy", "ext 4200")]
    print("\nChain hops as graph links (either direction):")
    for qid, a, b in hops:
        rels = [f"{a if u == a else b} --{k}--> {b if u == a else a}" for u, v in ((a, b), (b, a))
                if g.has_node(u) and g.has_node(v) and g.has_edge(u, v) for k in g[u][v]]
        mark = "✅" if rels else "❌"
        print(f"  {mark} {qid}: {describe(g, a)}  ↔  {describe(g, b)}   {'; '.join(rels) if rels else 'no direct link'}")
        if not rels and a in g and b in g:
            try:
                path = nx.shortest_path(g.to_undirected(as_view=True), a, b)
                print(f"       path in {len(path) - 1} hops: {' → '.join(describe(g, n) for n in path)}")
            except nx.NetworkXNoPath:
                print("       no path at all")


def draw(g: nx.MultiDiGraph):
    """Draw the chain entities plus the links between them and a few neighbours (~20 nodes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [k for k in CHAIN_SEEDS + [node_key(n, "", "")[0] for n in CHAIN_NAMES] if k in g]
    keep = set(seeds)
    und = g.to_undirected(as_view=True)
    for a in seeds:  # the shortest path between every pair of seeds that are close (≤ 3 hops)
        for b in seeds:
            if a < b:
                try:
                    path = nx.shortest_path(und, a, b)
                    if len(path) <= 4:
                        keep |= set(path)
                except nx.NetworkXNoPath:
                    pass
    sub = g.subgraph(keep)
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(sub, seed=7, k=1.6)
    colors = ["#e07a5f" if n in seeds else "#a8c5da" for n in sub]
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=900)
    nx.draw_networkx_labels(sub, pos, {n: sub.nodes[n]["name"][:28] for n in sub}, font_size=8)
    nx.draw_networkx_edges(sub, pos, arrows=True, arrowsize=12, edge_color="#888", connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edge_labels(sub, pos, {(u, v): k for u, v, k in sub.edges(keys=True)}, font_size=6)
    plt.title("Chain questions' neighbourhood (orange = entities the chain questions name or need)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PICTURE_FILE, dpi=130)
    print(f"\npicture: {PICTURE_FILE.name} ({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges)")


def report(g: nx.MultiDiGraph, triples: dict, seconds: float):
    tin = sum(t["tokens_in"] for t in triples.values())
    tout = sum(t["tokens_out"] for t in triples.values())
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    n_triples = sum(len(t["triples"]) for t in triples.values())
    COST_FILE.write_text(json.dumps({"chunks": len(triples), "llm_calls": len(triples), "triples": n_triples,
                                     "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 4),
                                     "nodes": g.number_of_nodes(), "edges": g.number_of_edges()}, indent=1))
    print(f"\nIndexing: {len(triples)} LLM calls, {tin:,} tokens in / {tout:,} out = ${cost:.4f}"
          + (f", {seconds:.0f} s" if seconds else " (from cache)"))
    print(f"Graph: {n_triples} triples -> {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("Node types:", dict(Counter(d["type"] for _, d in g.nodes(data=True)).most_common()))
    print("Relations: ", dict(Counter(k for _, _, k in g.edges(keys=True)).most_common()))
    print("\nMost connected nodes (hubs):")
    for n, deg in sorted(g.degree, key=lambda x: -x[1])[:12]:
        print(f"  {deg:3}  {g.nodes[n]['name']}  ({g.nodes[n]['type']})")
    # Possible duplicates: two keys where one contains the other ("pharmacy" / "central pharmacy")
    keys = [k for k in g if not CODE.fullmatch(k)]
    dups = [(a, b) for a in keys for b in keys if a != b and len(a) > 4 and re.search(rf"\b{re.escape(a)}\b", b)]
    print(f"\nName variants that stayed separate nodes ({len(dups)}), first 15:")
    for a, b in dups[:15]:
        print(f"  '{a}'  /  '{b}'")


def main():
    chunks, notes = contextual().load_chunks(), contextual().load_notes()
    print(f"loaded: {len(chunks)} chunks from sprint 03, {len(name_to_code())} entry names -> codes from sprint 08")

    if "--sample" in sys.argv:
        needles = ["Monitoring: aPTT (LC-0625)", "LC-0625 Activated", "start Norvessa", "Vancorin with Norvessa",
                   "call pharmacy for the antidote", "Central pharmacy: extension 4200"]
        sample = [c for c in chunks if any(n in c.text for n in needles)]
        triples = extract_all(sample, notes)
        for c in sample:
            print(f"\n=== {c.chunk_id}  ({len(triples[c.chunk_id]['triples'])} triples)")
            for t in triples[c.chunk_id]["triples"]:
                hk, tk = node_key(t["head"], t["head_type"], c.doc_id)[0], node_key(t["tail"], t["tail_type"], c.doc_id)[0]
                print(f"  {t['head']} ({t['head_type']})  --{t['relation']}-->  {t['tail']} ({t['tail_type']})"
                      f"      keys: {hk} | {tk}")
        return

    start = time.time()
    before = len(json.loads(TRIPLES_FILE.read_text())) if TRIPLES_FILE.exists() else 0
    triples = extract_all(chunks, notes)
    seconds = time.time() - start if len(triples) > before else 0
    triples = {c.chunk_id: triples[c.chunk_id] for c in chunks}
    extra = code_links(chunks, notes)
    print(f"code links (regex, free): {sum(map(len, extra.values()))} from {len(extra)} chunks")
    g = build(chunks, {cid: t | {"triples": t["triples"] + extra.get(cid, [])} for cid, t in triples.items()})
    save(g)
    report(g, triples, seconds)
    chain_check(g)
    draw(g)


if __name__ == "__main__":
    main()
