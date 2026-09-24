import json, sys
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW, GRAPH

# ── Parameters: tune here if the graph is too large or too sparse ──
NGRAM     = (1, 2)
MIN_DF    = 5
MAX_DF    = 0.3       # drop terms appearing in > 30% of documents
TOP_TERMS = 400       # keep the N most distinctive terms as nodes
MIN_COOCC = 4         # a pair must co-occur in >= 4 documents
MIN_NPMI  = 0.15      # keep an edge only if its normalized PMI >= this

past = json.load(open(RAW / "past_works.json"))

# Title carries strong signal, so weight it by repeating it twice
docs = [((w["title"] or "") + ". ") * 2 + (w["abstract"] or "") for w in past]

# Generic academic / RL-boilerplate words that only add noise as nodes
custom_stop = {
    "learn", "learns", "learning", "learned", "method", "methods", "model",
    "models", "approach", "approaches", "using", "use", "used", "uses", "based",
    "paper", "results", "result", "problem", "problems", "propose", "proposed",
    "algorithm", "algorithms", "task", "tasks", "performance", "state", "states",
    "art", "novel", "new", "deep", "network", "networks", "neural", "training",
    "train", "trained", "function", "functions", "data", "work", "works",
    "different", "present", "presented", "demonstrate", "demonstrated", "show",
    "shown", "framework", "frameworks", "achieve", "achieved", "provide",
    "study", "studies", "experiment", "experiments", "experimental",
    "set", "given", "number", "large", "high", "low", "time", "times",
}
stop_words = list(ENGLISH_STOP_WORDS | custom_stop)
token_pattern = r"(?u)\b[a-z][a-z\-]{2,}\b"

# 1) Select nodes by TF-IDF (distinctiveness), not raw document frequency
tfidf = TfidfVectorizer(
    ngram_range=NGRAM, min_df=MIN_DF, max_df=MAX_DF,
    stop_words=stop_words, token_pattern=token_pattern, sublinear_tf=True,
)
Xtf = tfidf.fit_transform(docs)
vocab = np.array(tfidf.get_feature_names_out())
term_score = np.asarray(Xtf.sum(axis=0)).ravel()      # summed TF-IDF per term
keep = np.argsort(term_score)[::-1][:TOP_TERMS]
keep.sort()
terms = vocab[keep]

# 2) Rebuild a binary presence matrix restricted to the kept terms
counts = CountVectorizer(
    vocabulary=list(terms), ngram_range=NGRAM,
    token_pattern=token_pattern, stop_words=stop_words,
)
B = counts.fit_transform(docs)
B.data[:] = 1
doc_freq = np.asarray(B.sum(axis=0)).ravel()
N = B.shape[0]

# 3) Co-occurrence counts via matrix product (diagonal = doc freq)
C = (B.T @ B).tocoo()

# 4) Keep edges by co-occurrence count AND normalized PMI.
#    nPMI filters out pairs that co-occur only because both terms are common.
G = nx.Graph()
for i, t in enumerate(terms):
    G.add_node(t, freq=int(doc_freq[i]))

eps = 1e-12
for i, j, c in zip(C.row, C.col, C.data):
    if i >= j or c < MIN_COOCC:
        continue
    p_ij = c / N
    p_i = doc_freq[i] / N
    p_j = doc_freq[j] / N
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    npmi = pmi / (-np.log(p_ij + eps))
    if npmi >= MIN_NPMI:
        G.add_edge(terms[i], terms[j], weight=int(c), npmi=round(float(npmi), 3))

# Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
top_deg = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)[:15]
print("Top 15 connected terms:", [t for t, _ in top_deg])

nx.write_gexf(G, GRAPH / "tkg_v2.gexf")
print("Saved:", GRAPH / "tkg_v2.gexf")