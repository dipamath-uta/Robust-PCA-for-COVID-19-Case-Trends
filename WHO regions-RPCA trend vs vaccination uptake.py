import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

# Load data
df = pd.read_csv(os.path.join(BASE, "clean_weekly_with_100k.csv"), encoding="cp1252")
df["date"] = pd.to_datetime(df["date"])

L_who = pd.read_csv(
    os.path.join(BASE, "rpca_who_cases_per100k_lowrank.csv"),
    index_col=0, parse_dates=True
)

# Build WHO-region vaccination time series
vax_who = (df.groupby(["date", "who_region"])["pfv_per_hundred"]
             .mean()
             .unstack("who_region")
             .sort_index())

common_dates = L_who.index.intersection(vax_who.index)
L_who = L_who.loc[common_dates]
vax_who = vax_who.loc[common_dates]

regions = L_who.columns
n = len(regions)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False)

for i, r in enumerate(regions):
    ax = axes[i // ncols][i % ncols]

    ax.plot(L_who.index, L_who[r], color="blue", label="PCP cases per 100k")
    ax.set_ylabel("Cases per 100k (PCP)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.grid(True)

    ax2 = ax.twinx()
    if r in vax_who.columns:
        ax2.plot(vax_who.index, vax_who[r], color="orange", label="PfV per 100")
    ax2.set_ylabel("PfV per 100", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax.set_title(r)
    ax.set_xlabel("Date")

    if i == 0:
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8)

for j in range(i+1, nrows*ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("WHO regions – PCP trend of cases vs vaccination uptake", y=1.02, fontsize=14)
plt.tight_layout()
out_path = os.path.join(BASE, "who_cases_pcp_vs_vax.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)
plt.show()
