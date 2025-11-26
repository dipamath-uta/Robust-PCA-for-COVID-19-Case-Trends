import pandas as pd

# Load your dataset
df = pd.read_csv("merged_clean_weekly.csv")

# Drop rows where both new_cases and pfv_per_hundred are either 0 or missing
df = df[~(((df['new_cases'].isna()) | (df['new_cases'] == 0)) &
          ((df['pfv_per_hundred'].isna()) | (df['pfv_per_hundred'] == 0)))]

# Save the cleaned dataset
df.to_csv("clean_weekly_filtered.csv", index=False)



import pandas as pd

# Load dataset safely
merged_clean = pd.read_csv(
    "C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered.csv",
    encoding="cp1252"  # or use 'latin1' if needed
)

# Step 1: Drop rows where both new_cases and pfv_per_hundred are 0 or missing
merged_clean = merged_clean[
    ~(((merged_clean['new_cases'].isna()) | (merged_clean['new_cases'] == 0)) &
      ((merged_clean['pfv_per_hundred'].isna()) | (merged_clean['pfv_per_hundred'] == 0)))
]

# Step 2: Drop rows where pfv_per_hundred is empty (missing)
merged_clean = merged_clean.dropna(subset=['pfv_per_hundred'])

# Step 3 (optional): If some are zeros but not missing, remove those too
merged_clean = merged_clean[merged_clean['pfv_per_hundred'] != 0]

# Save cleaned data
merged_clean.to_csv(
    "C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv",
    index=False
)

print("✅ Cleaning complete! Saved as clean_weekly_filtered.csv")


import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv")

# Ensure date is recognized as datetime
df['date'] = pd.to_datetime(df['date'])

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['new_cases'], label='New Cases', linewidth=2)
plt.plot(df['date'], df['pfv_per_hundred'], label='PfV per Hundred', linewidth=2)

plt.title('COVID-19 Weekly: New Cases vs Vaccinations')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Compute new cases per hundred
df['new_cases_per_hundred'] = (df['new_cases'] / df['population']) * 100

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['new_cases_per_hundred'], label='New Cases per Hundred', linewidth=2)
plt.plot(df['date'], df['pfv_per_hundred'], label='PfV per Hundred', linewidth=2, color='orange')

plt.title('COVID-19 Weekly: New Cases vs Vaccination Rate (per 100 people)')
plt.xlabel('Date')
plt.ylabel('Per 100 People')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# --- Load (adjust path if needed) ---
df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv", encoding="cp1252")

# --- Prepare ---
df['date'] = pd.to_datetime(df['date'])
# make comparable scale
df['new_cases_per_hundred'] = (df['new_cases'] / df['population']) * 100

# (optional) small smoothing to reduce spikes
# df['new_cases_per_hundred'] = df.sort_values('date').groupby('country')['new_cases_per_hundred'].transform(lambda s: s.rolling(3, min_periods=1).mean())
# df['pfv_per_hundred'] = df.sort_values('date').groupby('country')['pfv_per_hundred'].transform(lambda s: s.rolling(3, min_periods=1).mean())

def plot_by_group(data, group_col, title_prefix, save_path=None):
    # drop rows missing the grouping value
    d = data.dropna(subset=[group_col]).copy()

    # aggregate by date + group (mean across countries in the group for that date)
    agg = (d.groupby([group_col, 'date'], as_index=False)
             [['new_cases_per_hundred', 'pfv_per_hundred']].mean())

    groups = agg[group_col].dropna().unique()
    groups = sorted(groups)

    # layout for subplots
    n = len(groups)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False, sharex=False, sharey=False)

    for i, g in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        sub = agg[agg[group_col] == g].sort_values('date')

        ax.plot(sub['date'], sub['new_cases_per_hundred'], label='New Cases per Hundred', linewidth=2)
        ax.plot(sub['date'], sub['pfv_per_hundred'], label='PfV per Hundred', linewidth=2)

        ax.set_title(f"{g}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Per 100 People')
        ax.grid(True)
        if i == 0:
            ax.legend()

    # hide any empty panels
    for j in range(i+1, nrows*ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(f"{title_prefix}: New Cases vs Vaccination (per 100 people)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

# --- 1) By Continent ---
plot_by_group(
    df,
    group_col='continent',
    title_prefix='By Continent',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/plot_by_continent.png"
)

# --- 2) By WHO Region ---
# If your column is named differently (e.g., 'who_region' or 'who_region_code'), set it here:
plot_by_group(
    df,
    group_col='who_region',   # change if your column name differs
    title_prefix='By WHO Region',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/plot_by_who_region.png"
)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv", encoding="cp1252")
df['date'] = pd.to_datetime(df['date'])

# metrics
df['new_cases_per_100k'] = (df['new_cases'] / df['population']) * 100_000

# (optional) smooth a bit
df = df.sort_values('date')
df['new_cases_per_100k']  = df.groupby('country')['new_cases_per_100k'].transform(lambda s: s.rolling(3, min_periods=1).mean())
df['pfv_per_hundred']     = df.groupby('country')['pfv_per_hundred'].transform(lambda s: s.rolling(3, min_periods=1).mean())

def plot_by_group_dual_axis(data, group_col, title_prefix, save_path=None):
    d = data.dropna(subset=[group_col]).copy()
    # mean across countries in group for each date
    agg = (d.groupby([group_col, 'date'], as_index=False)
             [['new_cases_per_100k', 'pfv_per_hundred']].mean())

    groups = sorted(agg[group_col].dropna().unique())
    n = len(groups)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), squeeze=False)

    for i, g in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        sub = agg[agg[group_col] == g].sort_values('date')

        # left axis: vaccination per 100
        ax.plot(sub['date'], sub['pfv_per_hundred'], label='PfV per 100', linewidth=2)
        ax.set_ylabel('Per 100 people')
        ax.grid(True)
        ax.set_title(g)

        # right axis: cases per 100k
        ax2 = ax.twinx()
        ax2.plot(sub['date'], sub['new_cases_per_100k'], label='New cases per 100k', linewidth=2, linestyle='--')
        ax2.set_ylabel('New cases per 100k')
        # (optional) if spikes dominate, try a log scale:
        # ax2.set_yscale('log')

        # single legend combining both axes (only once)
        if i == 0:
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

        ax.set_xlabel('Date')

    # hide empty panels
    for j in range(i+1, nrows*ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(f"{title_prefix}: Vaccination vs New Cases", y=1.02, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# By continent
plot_by_group_dual_axis(
    df, group_col='continent',
    title_prefix='By Continent',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/continent_dual_axis.png"
)

# By WHO region (rename if your column differs)
plot_by_group_dual_axis(
    df, group_col='who_region',
    title_prefix='By WHO Region',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/who_dual_axis.png"
)

