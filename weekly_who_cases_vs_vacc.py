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

# Detect time column again
if 'week' in df.columns:
    time_col = 'week'
    df['week'] = pd.to_datetime(df['week'])
elif 'date' in df.columns:
    time_col = 'date'
    df['date'] = pd.to_datetime(df['date'])
else:
    raise ValueError("Need a 'week' or 'date' column in clean_weekly_MEAN_latest.csv")

required = [time_col, 'who_region', 'new_cases_per_100k', 'pfv_per_hundred']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=['who_region'])

# Aggregate to WHO region x week
weekly_who = (
    df.groupby(['who_region', time_col], as_index=False)
      .agg({
          'new_cases_per_100k': 'mean',
          'pfv_per_hundred': 'mean'
      })
      .sort_values(time_col)
)

# ----------------------------------------------------
# Plot: weekly cases vs vaccination (WHO regions)
# ----------------------------------------------------
regions = weekly_who['who_region'].unique()
n = len(regions)
ncols = 3 if n >= 3 else n
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6 * ncols, 4.5 * nrows),
                         squeeze=False)

for i, reg in enumerate(regions):
    ax = axes[i // ncols][i % ncols]
    sub = weekly_who[weekly_who['who_region'] == reg]

    # Left axis: weekly mean cases per 100k
    ax.plot(sub[time_col], sub['new_cases_per_100k'],
            color="#1E90FF", linewidth=1.8,
            label="Weekly mean cases per 100k")
    ax.set_title(reg, fontsize=12)
    ax.set_ylabel("Cases per 100k", color="#1E90FF")
    ax.tick_params(axis="y", labelcolor="#1E90FF")

    # Right axis: vaccination (% fully vaccinated)
    ax2 = ax.twinx()
    ax2.plot(sub[time_col], sub['pfv_per_hundred'],
             color="#008000", linewidth=1.8,
             label="% fully vaccinated (weekly mean)")
    ax2.set_ylabel("% fully vaccinated", color="#008000")
    ax2.tick_params(axis="y", labelcolor="#008000")
    ax2.yaxis.set_label_coords(1.10, 0.5)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

# Turn off empty subplots
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")

fig.suptitle("Weekly Mean COVID-19 Cases vs Vaccination (WHO Regions)",
             fontsize=16, y=1.02)

plt.tight_layout()
out_path = os.path.join(BASE, "weekly_who_cases_vs_vaccination.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)
plt.show()




















