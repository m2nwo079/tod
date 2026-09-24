import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE, RESULTS

MIN_PAST = 100

def load_jsonl(name):
    return {r["concept_id"]: r for r in
            (json.loads(l) for l in open(EXCHANGE / name))}

qwen = load_jsonl("judged_qwen.jsonl")
gemini = load_jsonl("judged_gemini.jsonl")
diffusion = {r["concept_id"]: r for r in
             json.loads((RESULTS / "diffusion.json").read_text())["concepts"]}

ids = [c for c in qwen if c in gemini and c in diffusion]
rows = []
for c in ids:
    rows.append({
        "concept_id": c,
        "qwen": qwen[c]["median_score"],
        "gemini": gemini[c]["median_score"],
        "rel_growth": diffusion[c]["rel_growth"],
        "past_n": diffusion[c]["past_n"],
        "reliable": diffusion[c]["past_n"] >= MIN_PAST,
    })

rows.sort(key=lambda r: r["rel_growth"], reverse=True)

print(f"{'concept':<8}{'qwen':>6}{'gemini':>8}{'rel_grw':>9}{'past_n':>8}{'ok':>4}")
for r in rows:
    flag = "" if r["reliable"] else " *"
    print(f"{r['concept_id']:<8}{r['qwen']:>6.0f}{r['gemini']:>8.0f}"
          f"{r['rel_growth']:>9.2f}{r['past_n']:>8}{flag:>4}")
print("  * = low confidence (few past works)\n")

def corr(subset, label):
    q = [r["qwen"] for r in subset]
    g = [r["gemini"] for r in subset]
    d = [r["rel_growth"] for r in subset]
    rq, pq = spearmanr(q, d)
    rg, pg = spearmanr(g, d)
    print(f"--- {label} (n={len(subset)}) ---")
    print(f"Qwen   vs diffusion: rho={rq:+.3f} (p={pq:.3f})")
    print(f"Gemini vs diffusion: rho={rg:+.3f} (p={pg:.3f})")

corr(rows, "All concepts")
print()
corr([r for r in rows if r["reliable"]], "Reliable only")

out = RESULTS / "three_way.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"\nSaved -> {out}")