"""RAPTOR lite, the collapsed tree: every node of the tree in ONE Qdrant collection, raptor_medical.

  180 leaves     embedded from note + chunk, exactly as sprint 03 (payload text = the original chunk)
   22 summaries  embedded from the summary itself (payload doc_id = "raptor:L1-05", "raptor:L2-03", ...)

Indexing reuses sprint 03's index() unchanged: dense (bge) + BM25 vectors per point. Search then treats a summary
like any chunk, so one hybrid search picks whichever level fits the question best: the paper's "collapsed tree".
Free (local embeddings). Needs tree_medical.json (build_tree.py), sprint 03's notes, and Qdrant running.

Run from rag_sprints/:  uv run week2/10_raptor/index_tree.py
"""

import json
from types import SimpleNamespace

from build_tree import TREE
from thematic import ROOT, _load

COLLECTION = "raptor_medical"
CTX_DIR = ROOT / "week1" / "03_contextual_rag"


def main():
    ingest = _load("contextual_ingest", CTX_DIR / "ingest.py")
    tree, notes = json.loads(TREE.read_text()), ingest.load_notes()
    leaves = [SimpleNamespace(chunk_id=n["id"], doc_id=n["doc_id"], text=n["text"]) for n in tree["leaves"]]
    summaries = [SimpleNamespace(chunk_id=n["id"], doc_id=n["doc_id"], text=n["text"]) for n in tree["summaries"]]
    texts = [ingest.contextualized_text(notes[c.chunk_id]["note"], c.text) for c in leaves] + [s.text for s in summaries]
    print(f"indexing {len(leaves)} leaves + {len(summaries)} summaries into '{COLLECTION}'")
    ingest.index(COLLECTION, leaves + summaries, texts, None)


if __name__ == "__main__":
    main()
