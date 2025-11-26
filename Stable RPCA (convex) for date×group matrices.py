# ===== Compare RPCA methods for continents =====
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

# --- Helper to load if exists ---
def load_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"⚠️  Missing file, skipping: {os.path.basename(path)}")
        return None

# Original matrix
M = load_if_exists(os.path.join(BASE, "rpca_continent_cases_per100k_matrix.csv"))

# Plain RPCA / PCP (from the earlier robust_pca)
L_pcp = load_if_exists(os.path.join(BASE, "rpca_continent_cases_per100k_lowrank.csv"))
S_pcp = load_if_exists(os.path.join(BASE, "rpca_continent_cases_per100k_sparse.csv"))

# Stable RPCA
L_st = load_if_exists(os.path.join(BASE, "continent_cases_per100k_stable_lowrank.csv"))
S_st = load_if_exists(os.path.join(BASE, "continent_cases_per100k_stable_sparse.csv"))

# IRLS-RPCA
L_irls = load_if_exists(os.path.join(BASE, "continent_cases_per100k_irls_lowrank.csv"))
S_irls = load_if_exists(os.path.join(BASE, "continent_cases_per100k_irls_sparse.csv"))

# basic sanity
if M is None:
    raise RuntimeError("Original continent matrix not found. Cannot plot.")

cols = M.columns
M = M.sort_index()
if L_pcp is not None:  L_pcp = L_pcp[cols].sort_index()
if S_pcp is not None:  S_pcp = S_pcp[cols].sort_index()
if L_st is not None:   L_st = L_st[cols].sort_index()
if S_st is not None:   S_st = S_st[cols].sort_index()
if L_irls is not None: L_irls = L_irls[cols].sort_index()
if S_irls is not None: S_irls = S_irls[cols].sort_index()

groups = cols
n = len(groups)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False, sharex=True)

for i, g in enumerate(groups):
    ax = axes[i // ncols][i % ncols]

    # Original
    ax.plot(M.index, M[g], color="gray", alpha=0.5, label="Original")

    # Plain RPCA / PCP
    if L_pcp is not None:
        ax.plot(L_pcp.index, L_pcp[g], color="orange", linewidth=2, label="PCP Low-rank")

    # Stable RPCA
    if L_st is not None:
        ax.plot(L_st.index, L_st[g], color="green", linewidth=2, linestyle="--", label="Stable Low-rank")

    # IRLS-RPCA
    if L_irls is not None:
        ax.plot(L_irls.index, L_irls[g], color="red", linewidth=1.5, linestyle=":", label="IRLS Low-rank")

    ax.set_title(g)
    ax.grid(True)
    if i % ncols == 0:
        ax.set_ylabel("New cases per 100k")
    ax.set_xlabel("Date")
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)

# hide empty axes
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Continents — Comparison of RPCA Methods (Low-rank trends)", y=1.02, fontsize=14)
plt.tight_layout()
out_png = os.path.join(BASE, "rpca_continents_methods_comparison.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"✅ Saved: {out_png}")
plt.show()
