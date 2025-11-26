# ===== RPCA plots for WHO regions =====
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

M_path = os.path.join(BASE, "rpca_who_cases_per100k_matrix.csv")
L_path = os.path.join(BASE, "rpca_who_cases_per100k_lowrank.csv")
S_path = os.path.join(BASE, "rpca_who_cases_per100k_sparse.csv")

M = pd.read_csv(M_path, index_col=0, parse_dates=True)
L = pd.read_csv(L_path, index_col=0, parse_dates=True)
S = pd.read_csv(S_path, index_col=0, parse_dates=True)

cols = [c for c in M.columns if c in L.columns and c in S.columns]
M, L, S = M[cols], L[cols], S[cols]
M = M.sort_index(); L = L.sort_index(); S = S.sort_index()

groups = cols
n = len(groups)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), squeeze=False, sharex=True)

for i, g in enumerate(groups):
    ax = axes[i // ncols][i % ncols]
    ax.plot(M.index, M[g], color="gray", alpha=0.5, label="Original")
    ax.plot(L.index, L[g], color="orange", linewidth=2, label="Low-Rank (trend)")
    ax.plot(S.index, S[g], color="red", linewidth=1, label="Sparse (anomalies)")
    ax.set_title(g)
    ax.grid(True)
    if i % ncols == 0:
        ax.set_ylabel("New cases per 100k")
    ax.set_xlabel("Date")
    if i == 0:
        ax.legend(loc="upper right")

for j in range(i+1, nrows*ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("RPCA Decomposition — WHO Regions (new cases per 100k)", y=1.02, fontsize=14)
plt.tight_layout()
out_png = os.path.join(BASE, "rpca_who_side_by_side.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved: {out_png}")
plt.show()
