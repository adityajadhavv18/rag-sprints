"""RAPTOR lite, indexing path: group similar chunks, summarise each group, then group and summarise the summaries.

  level 0  180 leaves = sprint 03's chunks (their dense vectors are read straight from contextual_medical)
  level 1  KMeans(18) on the leaf vectors   -> one gpt-4o-mini summary per cluster
  level 2  KMeans(4)  on the summary vectors -> one summary per cluster of summaries

The paper (Sarthi et al., 2024) clusters with a Gaussian mixture over UMAP-reduced vectors, so a chunk can sit in two
clusters, and repeats until one cluster is left. Lite: plain KMeans (each chunk in exactly one cluster), 2 levels.
The summaries read the ORIGINAL chunk text (no sprint-03 notes), each labelled with its document.

Run from rag_sprints/:
  uv run week2/10_raptor/build_tree.py            # build + save tree_medical.json + index_cost.json (~$0.02)
                                                  # (if the tree exists: print it, free; --rebuild pays again)
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from eval_harness import settings
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from sklearn.cluster import KMeans

os.environ.setdefault("RAG_CORPUS", "medical")
HERE = Path(__file__).resolve().parent
TREE = HERE / "tree_medical.json"
INDEX_COST = HERE / "index_cost.json"
LEAF_COLLECTION = "contextual_medical"     # sprint 03
EMBED_MODEL = settings.embed_model         # bge-small, the same 384-dim embedder as every earlier sprint
K_LEVEL1, K_LEVEL2 = 18, 4                 # ~10 leaves per summary, ~4-5 summaries per top summary
PRICE_IN, PRICE_OUT = 0.15, 0.60           # gpt-4o-mini, $ per million tokens

# The paper's prompt is "Write a summary of the following, including as many key details as possible". Ours adds two
# things: say what the texts have in common (that's the point of a summary node), and keep names, codes and numbers,
# because those are what search matches. 300 words keeps a summary under bge's 512-token window.
# The last three rules came from the first build: it wrote "Corvatrel (HV-RX-2074)" (2074 is Corvetrol, the
# look-alike), listed "E-17 air in the line" without saying which pump (E-17 means something else on each), and ended
# every summary with "a comprehensive overview... emphasizing safety".
SUMMARY_PROMPT = """Below are {n} passages from one hospital's internal documents (formulary, lab catalog, protocols, \
policies, device manuals...). They were grouped together because they are about related things.

Write a summary of at most 300 words:
- Start with one sentence saying what these passages have in common and which documents they come from.
- Then cover the main rules, practices and themes they share.
- Keep the key details: drug and device names, codes (HV-RX-..., LC-..., CP-..., HP-..., E-..), numbers, extensions.
- Only use what the passages say.
- Write a code only next to the exact name it belongs to in the passages. Many names look alike (Corvetrol and \
Corvatrel are different drugs); copy each name letter by letter, and leave a code out rather than guess.
- Fault codes are reused across devices with different meanings: always name the device with each fault code.
- No closing sentence that only praises or sums up ("overall, these passages...").

{passages}"""


def load_leaves() -> list[dict]:
    """Sprint 03's 180 chunks with their dense vectors (note + chunk), in chunk order."""
    points, _ = QdrantClient("http://localhost:6333").scroll(LEAF_COLLECTION, limit=1000, with_payload=True,
                                                             with_vectors=["dense"])
    leaves = [{"id": p.payload["chunk_id"], "level": 0, "doc_id": p.payload["doc_id"], "text": p.payload["text"],
               "vector": p.vector["dense"]} for p in points]
    return sorted(leaves, key=lambda n: n["id"])


def cluster(nodes: list[dict], k: int) -> list[list[dict]]:
    """KMeans on the nodes' vectors (normalised, so it groups by cosine). Clusters sorted by their first member."""
    vectors = np.array([n["vector"] for n in nodes])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
    groups = [[n for n, label in zip(nodes, labels) if label == c] for c in range(k)]
    return sorted(groups, key=lambda g: g[0]["id"])


def summarise(group: list[dict], client: OpenAI) -> tuple[str, int, int]:
    passages = "\n\n".join(f"[{n['source']}]\n{n['text']}" for n in group)
    resp = client.chat.completions.create(
        model=settings.llm_model, temperature=0,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(n=len(group), passages=passages)}])
    return resp.choices[0].message.content.strip(), resp.usage.prompt_tokens, resp.usage.completion_tokens


def build_level(children: list[dict], k: int, level: int, client: OpenAI, embedder: TextEmbedding, cost: dict):
    """Cluster `children`, summarise each cluster, embed the summaries. Returns the new summary nodes."""
    nodes = []
    for i, group in enumerate(cluster(children, k), 1):
        for n in group:
            n["source"] = n["doc_id"] if n["level"] == 0 else f"summary {n['id']}"
        text, t_in, t_out = summarise(group, client)
        cost["llm_calls"] += 1
        cost["tokens_in"] += t_in
        cost["tokens_out"] += t_out
        leaves = [leaf for n in group for leaf in (n.get("leaves") or [n["id"]])]
        nodes.append({"id": f"L{level}-{i:02d}", "level": level, "doc_id": f"raptor:L{level}-{i:02d}", "text": text,
                      "children": [n["id"] for n in group], "leaves": leaves,
                      "docs": dict(Counter(leaf.split("::")[0] for leaf in leaves).most_common())})
        print(f"  {nodes[-1]['id']}: {len(group)} children, {t_in} tokens in", flush=True)
    for n, v in zip(nodes, embedder.passage_embed([n["text"] for n in nodes])):
        n["vector"] = v.tolist()
    return nodes


def build():
    client, embedder = OpenAI(), TextEmbedding(EMBED_MODEL)
    leaves = load_leaves()
    cost = {"leaves": len(leaves), "llm_calls": 0, "tokens_in": 0, "tokens_out": 0}
    start = time.time()
    print(f"level 1: {len(leaves)} leaves -> {K_LEVEL1} clusters")
    level1 = build_level(leaves, K_LEVEL1, 1, client, embedder, cost)
    print(f"level 2: {len(level1)} summaries -> {K_LEVEL2} clusters")
    level2 = build_level(level1, K_LEVEL2, 2, client, embedder, cost)

    cost |= {"level1_summaries": len(level1), "level2_summaries": len(level2),
             "cost_usd": round((cost["tokens_in"] * PRICE_IN + cost["tokens_out"] * PRICE_OUT) / 1e6, 4),
             "seconds": round(time.time() - start)}
    for n in leaves:
        n.pop("source", None)
    TREE.write_text(json.dumps({"leaves": [{k: v for k, v in n.items() if k != "vector"} for n in leaves],
                                "summaries": level1 + level2}, indent=1))
    INDEX_COST.write_text(json.dumps(cost, indent=1))


def report():
    tree = json.loads(TREE.read_text())
    texts = {n["id"]: n["text"] for n in tree["leaves"]}
    for s in tree["summaries"]:
        docs = ", ".join(f"{d} {c}" for d, c in s["docs"].items())
        print(f"\n{'=' * 100}\n{s['id']}  ({len(s['children'])} children, {len(s['leaves'])} leaves: {docs})")
        if s["level"] == 1:
            starts = [texts[c].split("\n")[0][:60] for c in s["children"]]
            print("  chunks start: " + " | ".join(starts))
        else:
            print("  children: " + ", ".join(s["children"]))
        print(f"\n{s['text']}")
    print(f"\n{'=' * 100}\nindex cost: {json.loads(INDEX_COST.read_text())}")


if __name__ == "__main__":
    if not TREE.exists() or "--rebuild" in sys.argv:
        build()
    report()
