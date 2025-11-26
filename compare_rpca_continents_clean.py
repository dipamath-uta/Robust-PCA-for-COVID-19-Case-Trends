# compare_rpca_continents_clean.py
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

M      = load(os.path.join(BASE, "rpca_continent_cases_per100k_matrix.csv"))
L_pcp  = load(os.path.join(BASE, "rpca_continent_cases_per100k_lowrank.csv"))
L_irls = load(os.path.join(BASE, "continent_cases_per100k_irls_lowrank.csv"))

cols = M.columns
M      = M[cols].sort_index()
L_pcp  = L_pcp[cols].sort_index()
L_irls = L_irls[cols].sort_index()

groups = cols
n = len(groups)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows),
                         squeeze=False, sharex=True)

for i, g in enumerate(groups):
    ax = axes[i // ncols][i % ncols]
    ax.plot(M.index, M[g], color="gray", alpha=0.5, label="Original")
    ax.plot(L_pcp.index, L_pcp[g], color="orange", linewidth=2, label="PCP low-rank")
    ax.plot(L_irls.index, L_irls[g], color="red", linewidth=1.5, linestyle=":",
            label="IRLS low-rank")

    ax.set_title(g)
    ax.grid(True)
    if i % ncols == 0:
        ax.set_ylabel("New cases per 100k")
    ax.set_xlabel("Date")
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)

for j in range(i+1, nrows*ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Continents – PCP vs IRLS RPCA (low-rank trends)", y=1.02, fontsize=14)
plt.tight_layout()
out_png = os.path.join(BASE, "rpca_continents_pcp_vs_irls.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)
plt.show()
