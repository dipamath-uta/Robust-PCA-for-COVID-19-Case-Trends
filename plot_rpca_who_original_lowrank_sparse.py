import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

# --------- Load the RPCA outputs for WHO regions ----------
M_path  = os.path.join(BASE, "rpca_who_cases_matrix.csv")
L_path  = os.path.join(BASE, "rpca_who_cases_lowrank.csv")
S_path  = os.path.join(BASE, "rpca_who_cases_sparse.csv")

M = load(M_path)   # original weekly new_cases_per_100k (matrix)
L = load(L_path)   # low-rank component
S = load(S_path)   # sparse component

# Align columns and index
cols = M.columns
M = M[cols].sort_index()
L = L[cols].sort_index()
S = S[cols].sort_index()

groups = cols
n = len(groups)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 3.8 * nrows),
    squeeze=False,
    sharex=True
)

# Use one colormap and pick light → medium → deep shades
cmap = plt.cm.Greens  # you can change to Blues, Reds, etc.

for i, g in enumerate(groups):
    ax = axes[i // ncols][i % ncols]

    # 3 shades from the colormap
    c_light = cmap(0.3)
    c_mid   = cmap(0.6)
    c_deep  = cmap(0.9)

    # Original signal
    ax.plot(
        M.index, M[g],
        color=c_light,
        linewidth=1.5,
        alpha=0.9,
        label="Original"
    )

    # Low-rank component
    ax.plot(
        L.index, L[g],
        color=c_mid,
        linewidth=2,
        label="Low-rank (L)"
    )

    # Sparse component
    ax.plot(
        S.index, S[g],
        color=c_deep,
        linewidth=1.5,
        linestyle="--",
        label="Sparse (S)"
    )

    ax.set_title(g)
    ax.grid(True, linestyle="--", alpha=0.4)

    if i % ncols == 0:
        ax.set_ylabel("Weekly new cases per 100k")

    ax.set_xlabel("Week")

    if i == 0:
        ax.legend(fontsize=8, loc="upper right")

# Turn off unused axes if grid not full
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

# ----- Date axis formatting -----
for row in axes:
    for ax in row:
        if ax.has_data():
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

fig.suptitle(
    "RPCA decomposition by WHO region: Original vs Low-rank vs Sparse",
    y=1.02,
    fontsize=14
)
plt.tight_layout()

out_png = os.path.join(BASE, "rpca_who_original_lowrank_sparse.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)

plt.show()
