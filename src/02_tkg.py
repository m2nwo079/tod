import json, sys
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW, GRAPH

# ── Parameters: tune here if the graph is too large or too sparse ──
NGRAM = (1, 2)
MIN_DF = 5
MAX_DF = 0.4
TOP_TERMS = 800
MIN_COOCC = 3

past = json.load(open(RAW / "past_works.json"))

# Title carries strong signal, so weight it by repeating it twice
docs = [((w["title"] or "") + ". ") * 2 + (w["abstract"] or "") for w in past]

# Domain-generic words that only add noise as graph nodes
custom_stop = {
    "learning", "method", "methods", "model", "models", "approach", "using",
    "based", "paper", "results", "result", "problem", "problems", "propose",
    "proposed", "algorithm", "algorithms", "task", "tasks", "performance",
    "state", "art", "novel", "new", "deep", "network", "networks", "neural",
    "training", "train", "trained", "function", "functions", "data",
}
stop_words = list(ENGLISH_STOP_WORDS | custom_stop)

vectorizer = CountVectorizer(
    ngram_range = NGRAM,
    min_df = MIN_DF,
    max_df = MAX_DF,
    stop_words=stop_words,
    token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b",
)
X = vectorizer.fit_transform(docs)
vocab = np.array(vectorizer.get_feature_names_out())

# Binary presence matrix (doc x term), avoiding the sparse-compare warning
B = X.copy()
B.data[:] = 1

# Keep only the TOP_TERMS most frequent terms as nodes
doc_freq = np.asarray(B.sum(axis=0)).ravel()
keep = np.argsort(doc_freq)[::-1][:TOP_TERMS]
keep.sort()
terms = vocab[keep]
Bk = B[:, keep]

# Term-term co-occurrence via matrix product (diagonal = doc freq)
C = (Bk.T @ Bk).tocoo()

G = nx.Graph()
for i, t in enumerate(terms):
    G.add_node(t, freq=int(doc_freq[keep[i]]))

# Upper triangle only; skip the diagonal; apply the co-occurrence threshold
for i, j, c in zip(C.row, C.col, C.data):
    if i < j and c >= MIN_COOCC:
        G.add_edge(terms[i], terms[j], weight=int(c))

# Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
top_deg = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:15]
print("Top 15 connected terms:", [t for t, _ in top_deg])

nx.write_gexf(G, GRAPH / "tkg.gexf")
print("Saved:", GRAPH / "tkg.gexf")