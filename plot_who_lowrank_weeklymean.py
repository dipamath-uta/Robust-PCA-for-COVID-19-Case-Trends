import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

# --- Load matrices (weekly MEAN) ---
M      = load(os.path.join(BASE, "rpca_who_cases_matrix.csv"))          # original weekly mean
L_pcp  = load(os.path.join(BASE, "rpca_who_cases_lowrank.csv"))         # convex RPCA (PCP)
L_irls = load(os.path.join(BASE, "weekly_who_cases_per100k_ircur_lowrank.csv")) # IRCUR 2021

cols = M.columns
M      = M[cols].sort_index()
L_pcp  = L_pcp[cols].sort_index()
L_irls = L_irls[cols].sort_index()

regions = list(cols)
n = len(regions)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 3.8 * nrows),
    squeeze=False,
    sharex=True
)

for i, r in enumerate(regions):
    ax = axes[i // ncols][i % ncols]

    ax.plot(M.index, M[r],
            color="#008000", linewidth=2.2, label="Original")
    
    ax.plot(L_irls.index, L_irls[r],
            color="#4B0082", linewidth=2.0, label="Non-Convex RPCA (IRCUR)")
    ax.plot(L_pcp.index, L_pcp[r],
            color="#FF8C00", linewidth=1.8, label="Convex RPCA (PCP)")

    ax.set_title(r)
    ax.grid(True, linestyle="--", alpha=0.3)

    if i % ncols == 0:
        ax.set_ylabel("Weekly new cases per 100k")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    if i == 0:
        ax.legend(fontsize=8, loc="upper right")

for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle(
    "WHO regions – Original vs Convex RPCA vs Non-Convex RPCA\n(weekly MEAN cases per 100k)",
    y=1.02,
    fontsize=14
)
plt.tight_layout()

out_png = os.path.join(BASE, "who_lowrank_weeklymean_pcp_ircur.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Saved:", out_png)

plt.show()
