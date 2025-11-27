import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
WEEKLY_FILE = os.path.join(BASE, "clean_weekly_MEAN_latest.csv")

# ----------------------------------------------------
# Load weekly mean data
# ----------------------------------------------------
df = pd.read_csv(WEEKLY_FILE)

# Try to detect the time column: 'week' or 'date'
if 'week' in df.columns:
    time_col = 'week'
    df['week'] = pd.to_datetime(df['week'])
elif 'date' in df.columns:
    time_col = 'date'
    df['date'] = pd.to_datetime(df['date'])
else:
    raise ValueError("Need a 'week' or 'date' column in clean_weekly_MEAN_latest.csv")

required = [time_col, 'continent', 'new_cases_per_100k', 'pfv_per_hundred']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Drop rows without continent
df = df.dropna(subset=['continent'])

# Aggregate to continent x week (mean of all countries in that continent)
weekly_cont = (
    df.groupby(['continent', time_col], as_index=False)
      .agg({
          'new_cases_per_100k': 'mean',
          'pfv_per_hundred': 'mean'
      })
      .sort_values(time_col)
)

# ----------------------------------------------------
# Plot: weekly cases vs vaccination (continents)
# ----------------------------------------------------
continents = weekly_cont['continent'].unique()
n = len(continents)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 4.5 * nrows),
                         squeeze=False)

for i, cont in enumerate(continents):
    ax = axes[i // ncols][i % ncols]
    sub = weekly_cont[weekly_cont['continent'] == cont]

    # Left axis: weekly mean new cases per 100k (green)
    ax.plot(sub[time_col], sub['new_cases_per_100k'],
            color="#1E90FF", linewidth=1.8,
            label="Weekly mean cases per 100k")
    ax.set_title(cont, fontsize=12)
    ax.set_ylabel("Cases per 100k", color="#1E90FF")
    ax.tick_params(axis="y", labelcolor="#1E90FF")

    # Right axis: weekly mean vaccination (% fully vaccinated)
    ax2 = ax.twinx()
    ax2.plot(sub[time_col], sub['pfv_per_hundred'],
             color="#008000", linewidth=1.8,
             label="% fully vaccinated (weekly mean)")
    ax2.set_ylabel("% fully vaccinated", color="#008000")
    ax2.tick_params(axis="y", labelcolor="#008000")

    # Date formatting (6-month ticks)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

# Turn off any unused subplots
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Weekly Mean COVID-19 Cases vs Vaccination (Continents)",
             fontsize=16, y=1.02)

plt.tight_layout()
out_path = os.path.join(BASE, "weekly_continent_cases_vs_vaccination.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)
plt.show()
