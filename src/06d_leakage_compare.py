import sys, json
from pathlib import Path
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE, RESULTS

def load(name):
    return {r["concept_id"]: r["median_score"]
            for r in (json.loads(l) for l in open(EXCHANGE / name))}

q_orig  = load("judged_qwen.jsonl")
q_blind = load("judged_qwen_blind.jsonl")
g_orig  = load("judged_gemini.jsonl")
g_blind = load("judged_gemini_blind.jsonl")
diff = {r["concept_id"]: r for r in
        json.loads((RESULTS / "diffusion.json").read_text())["concepts"]}

ids = [c for c in diff if c in q_orig and c in g_orig
       and c in q_blind and c in g_blind]

# Print the full comparison table
print(f"{'id':<6}{'Q_orig':>7}{'Q_blind':>8}{'G_orig':>7}{'G_blind':>8}"
      f"{'rel_grw':>9}{'ok':>4}")
for c in sorted(ids, key=lambda x: diff[x]["rel_growth"], reverse=True):
    ok = "" if diff[c]["past_n"] >= 100 else " *"
    print(f"{c:<6}{q_orig[c]:>7}{q_blind[c]:>8}{g_orig[c]:>7}{g_blind[c]:>8}"
          f"{diff[c]['rel_growth']:>9.2f}{ok:>4}")

def rho(scores, subset, label):
    d = [diff[c]["rel_growth"] for c in subset]
    s = [scores[c] for c in subset]
    r, p = spearmanr(s, d)
    print(f"  {label:<16} rho={r:+.3f} (p={p:.3f})")

reliable = [c for c in ids if diff[c]["past_n"] >= 100]

for scope_name, subset in [("ALL (n=%d)" % len(ids), ids),
                           ("RELIABLE (n=%d)" % len(reliable), reliable)]:
    print(f"\n--- Correlation with diffusion, {scope_name} ---")
    rho(q_orig,  subset, "Qwen original")
    rho(q_blind, subset, "Qwen blind")
    rho(g_orig,  subset, "Gemini original")
    rho(g_blind, subset, "Gemini blind")

out = RESULTS / "leakage_compare.json"
out.write_text(json.dumps(
    {c: {"q_orig": q_orig[c], "q_blind": q_blind[c],
         "g_orig": g_orig[c], "g_blind": g_blind[c],
         "rel_growth": diff[c]["rel_growth"], "past_n": diff[c]["past_n"]}
     for c in ids}, ensure_ascii=False, indent=2))
print(f"\nSaved -> {out}")