import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

def load(name):
    return pd.read_csv(os.path.join(BASE, name), index_col=0, parse_dates=True)

M       = load("rpca_continent_cases_per100k_matrix.csv")   # original
S_pcp   = load("rpca_continent_cases_per100k_sparse.csv")   # convex sparse
S_irls  = load("daily_continent_cases_per100k_ircur_sparse.csv")   # IRCUR sparse

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

    ax.plot(M.index, M[g],
            color="#008000", linewidth=8.0, label="Original")
    
    ax.plot(S_pcp.index, S_pcp[g],
            color="#FF8C00", linewidth=3.0, label="Convex RPCA (PCP) sparse")
    ax.plot(S_irls.index, S_irls[g],
            color="#4B0082", linewidth=1.0, label="Non-Convex RPCA(IRCUR) sparse")
    
    ax.set_title(g)
    ax.grid(True, linestyle="--", alpha=0.3)

    if i % ncols == 0:
        ax.set_ylabel("Sparse component (daily)")

    # 6-month ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")

    if i == 0:
        ax.legend(fontsize=8, loc="upper right")

for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Continents – DAILY Original vs Convex(sparse) vs Non-Convex(sparse)",
             y=1.02, fontsize=14)
plt.tight_layout()

out = os.path.join(BASE, "continents_daily_sparse_pcp_ircur.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Saved:", out)

plt.show()
