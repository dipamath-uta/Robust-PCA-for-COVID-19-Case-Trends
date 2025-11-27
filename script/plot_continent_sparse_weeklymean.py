import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

# --- Load matrices (weekly MEAN) ---
M       = load(os.path.join(BASE, "rpca_continent_cases_matrix.csv"))           # original weekly mean
S_pcp   = load(os.path.join(BASE, "rpca_continent_cases_sparse.csv"))           # convex PCP sparse
S_irls  = load(os.path.join(BASE, "weekly_continent_cases_per100k_ircur_sparse.csv"))   # IRCUR sparse

# Align index & columns
cols = M.columns
M      = M[cols].sort_index()
S_pcp  = S_pcp[cols].sort_index()
S_irls = S_irls[cols].sort_index()

groups = list(cols)
n = len(groups)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 3.8 * nrows),
    squeeze=False,
    sharex=True
)

for i, g in enumerate(groups):
    ax = axes[i // ncols][i % ncols]

    # --- Required Color Sequence ---
    ax.plot(M.index, M[g],
            color="#008000", linewidth=2.2, label="Original")
    ax.plot(S_pcp.index, S_pcp[g],
            color="#FF8C00", linewidth=1.8, label="Convex RPCA (PCP)")
    ax.plot(S_irls.index, S_irls[g],
            color="#4B0082", linewidth=1.4, label="Non-convex RPCA (IRCUR)")

    

    ax.set_title(g)
    ax.grid(True, linestyle="--", alpha=0.3)

    if i % ncols == 0:
        ax.set_ylabel("Weekly sparse component")

    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    if i == 0:
        ax.legend(fontsize=8, loc="upper right")

# Turn off empty axes
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle(
    "Continents – Sparse Component: Original vs Convex RPCA vs Non-Convex RPCA\n(weekly MEAN cases)",
    y=1.02,
    fontsize=14
)

plt.tight_layout()

out_png = os.path.join(BASE, "continents_sparse_mean_pcp_ircur.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)

plt.show()
