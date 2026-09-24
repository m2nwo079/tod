import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE, RESULTS

def load(name):
    return {r["concept_id"]: r for r in
            (json.loads(l) for l in open(EXCHANGE / name))}

qwen = load("judged_qwen.jsonl")
gemini = load("judged_gemini.jsonl")

# Align on concept_id, preserving a stable order
ids = [c for c in qwen if c in gemini]
q = np.array([qwen[c]["median_score"] for c in ids], dtype=float)
g = np.array([gemini[c]["median_score"] for c in ids], dtype=float)

print(f"Concepts compared: {len(ids)}\n")
print(f"{'concept':<8}{'qwen':>6}{'gemini':>8}{'diff':>6}")
for c, qs, gs in zip(ids, q, g):
    print(f"{c:<8}{qs:>6.0f}{gs:>8.0f}{gs - qs:>+6.0f}")

# Agreement metrics (small-sample caveat applies)
rho, p_rho = spearmanr(q, g)
r, p_r = pearsonr(q, g)
mae = np.mean(np.abs(q - g))
exact = np.mean(q == g)

print("\n--- Agreement (n=7, interpret with caution) ---")
print(f"Spearman rho : {rho:.3f}  (p={p_rho:.3f})")
print(f"Pearson  r   : {r:.3f}  (p={p_r:.3f})")
print(f"Mean abs diff: {mae:.2f}")
print(f"Exact match  : {exact*100:.0f}%")

# Largest disagreements = qualitative discussion cases
print("\n--- Biggest disagreements ---")
order = np.argsort(-np.abs(g - q))
for i in order[:3]:
    print(f"  {ids[i]}: qwen={q[i]:.0f}, gemini={g[i]:.0f}, diff={g[i]-q[i]:+.0f}")

# Save a merged table for later steps (Step 6 backtest, Step 8 comparison)
RESULTS.mkdir(parents=True, exist_ok=True)
merged = [{"concept_id": c, "qwen": float(qwen[c]["median_score"]),
           "gemini": float(gemini[c]["median_score"])} for c in ids]
out = RESULTS / "judge_agreement.json"
out.write_text(json.dumps(
    {"concepts": merged,
     "spearman": rho, "pearson": r, "mae": mae, "exact_match": exact},
    ensure_ascii=False, indent=2))
print(f"\nSaved -> {out}")