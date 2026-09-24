import sys
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GRAPH, RESULTS

# ── Parameters ──
GRAPH_FILE = "tkg_v2.gexf"
TOP_N      = 60      # number of highest-strength nodes to show in the readable view

G = nx.read_gexf(GRAPH / GRAPH_FILE)
print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

fig_dir = RESULTS / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ── View 1: full graph, structure only ──
plt.figure(figsize=(14, 14))
pos_full = nx.spring_layout(G, k=0.15, seed=42)
nx.draw_networkx_edges(G, pos_full, alpha=0.06, width=0.4)
nx.draw_networkx_nodes(G, pos_full, node_size=15, node_color="#1f77b4", alpha=0.7)
plt.title(f"TKG v2 — full ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
plt.axis("off")
plt.tight_layout()
plt.savefig(fig_dir / "tkg_v2_full.png", dpi=150)
plt.close()
print("Saved:", fig_dir / "tkg_v2_full.png")

# ── View 2: top-N nodes by weighted degree, with labels ──
strength = dict(G.degree(weight="weight"))
top_nodes = sorted(strength, key=strength.get, reverse=True)[:TOP_N]
H = G.subgraph(top_nodes).copy()

plt.figure(figsize=(16, 16))
pos_h = nx.spring_layout(H, k=0.5, seed=42)
deg_h = dict(H.degree(weight="weight"))
sizes = [deg_h[n] * 0.8 for n in H.nodes()]
nx.draw_networkx_edges(H, pos_h, alpha=0.25, width=0.6)
nx.draw_networkx_nodes(H, pos_h, node_size=sizes, node_color="#ff7f0e", alpha=0.85)
nx.draw_networkx_labels(H, pos_h, font_size=9)
plt.title(f"TKG v2 — top {TOP_N} terms by connection strength")
plt.axis("off")
plt.tight_layout()
plt.savefig(fig_dir / "tkg_v2_top.png", dpi=150)
plt.close()
print("Saved:", fig_dir / "tkg_v2_top.png")