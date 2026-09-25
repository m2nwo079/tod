import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS

fig_dir = RESULTS / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Confirmed pipeline values
all_concepts = ["C05", "C06", "C02", "C00", "C04", "C01", "C03"]
qwen_all = [7, 8, 8, 8, 8, 7, 7]
gemini_all = [4, 8, 9, 8, 8, 7, 7]

reliable = ["C05", "C06", "C02", "C00"]
rel_growth = [5.23, 3.59, 3.06, 2.47]
gem_orig = [4, 8, 9, 8]
gem_blind = [8, 9, 3, 8]
dvf_total = [10, 8, 8, 9]

QWEN_C = "#8a94a0"
GEM_C = "#2f5d50"
TRUTH_C = "#a9761a"
PROMISE_C = "#b98a3a"


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=9)


# Figure 1: dual-judge scores across all 7 concepts
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(all_concepts))
w = 0.38
ax.bar(x - w / 2, qwen_all, w, label="Qwen3-8B", color=QWEN_C)
ax.bar(x + w / 2, gemini_all, w, label="Gemini", color=GEM_C)
ax.set_xticks(x)
ax.set_xticklabels([c + (" *" if c in ("C04", "C01", "C03") else "") for c in all_concepts])
ax.set_ylim(0, 9.5)
style(ax, "Dual-judge promise scores (1-9)", "median score")
plt.tight_layout()
plt.savefig(fig_dir / "dual_judge.png", dpi=150)
plt.close()

# Figure 2: leakage mitigation (Gemini original vs blind vs diffusion)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(reliable))
w = 0.27
ax.bar(x - w, gem_orig, w, label="Gemini original (1-9)", color=QWEN_C)
ax.bar(x, gem_blind, w, label="Gemini blind (1-9)", color=GEM_C)
ax.bar(x + w, rel_growth, w, label="Actual diffusion (rel.)", color=TRUTH_C)
ax.set_xticks(x)
ax.set_xticklabels(reliable)
style(ax, "Leakage mitigation vs actual diffusion (diffusion-sorted)", "score / diffusion")
plt.tight_layout()
plt.savefig(fig_dir / "leakage_mitigation.png", dpi=150)
plt.close()

# Figure 3: three-way comparison (LLM promise vs DVF vs diffusion)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(reliable))
w = 0.27
ax.bar(x - w, gem_orig, w, label="LLM promise (1-9)", color=PROMISE_C)
ax.bar(x, dvf_total, w, label="DVF total (3-15)", color=GEM_C)
ax.bar(x + w, rel_growth, w, label="Actual diffusion (rel.)", color=TRUTH_C)
ax.set_xticks(x)
ax.set_xticklabels(reliable)
style(ax, "Three-way comparison (different scales)", "respective scale")
plt.tight_layout()
plt.savefig(fig_dir / "three_way.png", dpi=150)
plt.close()

print("Saved:", fig_dir / "dual_judge.png")
print("Saved:", fig_dir / "leakage_mitigation.png")
print("Saved:", fig_dir / "three_way.png")