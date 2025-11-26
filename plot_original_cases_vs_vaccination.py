import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
DATA = os.path.join(BASE, "clean_weekly_SUM_latest.csv")

df = pd.read_csv(DATA, parse_dates=["week"])

# ---------------- Continent-level mean values ----------------
weekly = (
    df.groupby(["continent", "week"], as_index=False)
      .agg({
          "new_cases_per_100k": "sum",
          "pfv_per_hundred": "last",
          "population": "last"
      })
)

continents = weekly["continent"].unique()

fig, ax = plt.subplots(figsize=(12,6))

for cont in continents:
    sub = weekly[weekly["continent"] == cont]
    ax.plot(
        sub["week"],
        sub["new_cases_per_100k"],
        alpha=0.6,
        linewidth=2,
        label=f"{cont} – new cases"
    )

ax2 = ax.twinx()

for cont in continents:
    sub = weekly[weekly["continent"] == cont]
    ax2.plot(
        sub["week"],
        sub["pfv_per_hundred"],
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=f"{cont} – vaccination"
    )

# ----- Formatting -----
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_title("Original Weekly New Cases vs Vaccination Uptake – By Continent")
ax.set_ylabel("Weekly New Cases per 100k")
ax2.set_ylabel("Vaccination (% fully vaccinated)")

# Combine legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(BASE, "original_cases_vs_vaccination.png"), dpi=200)
plt.show()
