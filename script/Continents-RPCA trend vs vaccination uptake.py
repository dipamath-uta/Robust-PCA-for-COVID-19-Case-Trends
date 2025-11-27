import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"

# --- Load data ---
df = pd.read_csv(os.path.join(BASE, "clean_weekly_with_100k.csv"), encoding="cp1252")
df["date"] = pd.to_datetime(df["date"])

# PCP low-rank cases per 100k (continents)
L_cont = pd.read_csv(
    os.path.join(BASE, "rpca_continent_cases_per100k_lowrank.csv"),
    index_col=0, parse_dates=True
)

# --- Build continent-level vaccination series (PfV per 100) ---
# average PfV per 100 over countries in each continent for each date
vax_cont = (df.groupby(["date", "continent"])["pfv_per_hundred"]
              .mean()
              .unstack("continent")
              .sort_index())

# Align dates between L_cont and vax_cont
common_dates = L_cont.index.intersection(vax_cont.index)
L_cont = L_cont.loc[common_dates]
vax_cont = vax_cont.loc[common_dates]

continents = L_cont.columns
n = len(continents)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False)

for i, c in enumerate(continents):
    ax = axes[i // ncols][i % ncols]

    # left axis: PCP low-rank cases per 100k (blue)
    ax.plot(L_cont.index, L_cont[c], color="blue", label="PCP cases per 100k")
    ax.set_ylabel("Cases per 100k (PCP)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.grid(True)

    # right axis: vaccination per 100 (orange)
    ax2 = ax.twinx()
    if c in vax_cont.columns:
        ax2.plot(vax_cont.index, vax_cont[c], color="orange", label="PfV per 100")
    ax2.set_ylabel("PfV per 100", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax.set_title(c)
    ax.set_xlabel("Date")

    # Legend (only once)
    if i == 0:
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8)

# hide empty subplots if any
for j in range(i+1, nrows*ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Continents – PCP trend of cases vs vaccination uptake", y=1.02, fontsize=14)
plt.tight_layout()
out_path = os.path.join(BASE, "continents_cases_pcp_vs_vax.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)
plt.show()
