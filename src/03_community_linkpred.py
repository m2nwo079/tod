import json, sys
from pathlib import Path
import networkx as nx
import community as community_louvain

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW, GRAPH, EXCHANGE

# ── Parameters ──
GRAPH_FILE    = "tkg_v2.gexf"
RESOLUTION    = 1.0      # higher -> more, smaller communities
MIN_COMM_SIZE = 6        # ignore communities smaller than this
TOP_LINKS     = 8        # latent-edge candidates kept per community
ABS_PER_CONC  = 3        # representative abstracts attached per concept
SEED          = 42

G = nx.read_gexf(GRAPH / GRAPH_FILE)
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 1) Community detection (themes)
partition = community_louvain.best_partition(G, weight="weight",
                                             resolution=RESOLUTION, random_state=SEED)
communities = {}
for node, cid in partition.items():
    communities.setdefault(cid, []).append(node)
communities = {c: ns for c, ns in communities.items() if len(ns) >= MIN_COMM_SIZE}
print(f"Communities (size >= {MIN_COMM_SIZE}): {len(communities)}")

# 2) Link prediction within each community (Adamic-Adar on non-edges)
def top_latent_edges(nodes, k):
    sub = G.subgraph(nodes)
    non_edges = list(nx.non_edges(sub))
    if not non_edges:
        return []
    scored = nx.adamic_adar_index(sub, non_edges)
    ranked = sorted(scored, key=lambda x: x[2], reverse=True)
    return [(u, v, round(s, 3)) for u, v, s in ranked[:k]]

# 3) Load works once to attach representative abstracts by term overlap
works = json.load(open(RAW / "past_works.json"))
def rep_abstracts(terms, n):
    tset = set(terms)
    scored = []
    for w in works:
        text = ((w["title"] or "") + " " + (w["abstract"] or "")).lower()
        if not text.strip():
            continue
        hits = sum(1 for t in tset if t in text)
        if hits:
            scored.append((hits, w["title"], w["abstract"][:600]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"title": t, "abstract": a} for _, t, a in scored[:n]]

# 4) Build one concept candidate per community
concepts = []
for cid, nodes in communities.items():
    strength = G.subgraph(nodes).degree(weight="weight")
    core_terms = [t for t, _ in sorted(strength, key=lambda x: x[1], reverse=True)[:12]]
    latent = top_latent_edges(nodes, TOP_LINKS)
    concepts.append({
        "concept_id": f"C{cid:02d}",
        "community_id": cid,
        "size": len(nodes),
        "core_terms": core_terms,
        "latent_links": [f"{u} -- {v}" for u, v, _ in latent],
        "abstracts": rep_abstracts(core_terms, ABS_PER_CONC),
    })

concepts.sort(key=lambda c: c["size"], reverse=True)

# 5) Write the boundary file consumed later by the Colab LLM steps
out = EXCHANGE / "concepts.jsonl"
with open(out, "w") as f:
    for c in concepts:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Concepts written: {len(concepts)} -> {out}")
for c in concepts:
    print(f"  {c['concept_id']} (size {c['size']}): {', '.join(c['core_terms'][:6])}")