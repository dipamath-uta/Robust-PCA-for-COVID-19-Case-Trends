import pandas as pd
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_csv("C:/Users/dipac/Downloads/covid-vax-project/clean_weekly_filtered2.csv", encoding="cp1252")

# --- Prepare data ---
df['date'] = pd.to_datetime(df['date'])
df['new_cases_per_100k'] = (df['new_cases'] / df['population']) * 100_000  # cases per 100k

# Optional smoothing to reduce noise
df = df.sort_values('date')
df['new_cases_per_100k']  = df.groupby('country')['new_cases_per_100k'].transform(lambda s: s.rolling(3, min_periods=1).mean())
df['pfv_per_hundred']     = df.groupby('country')['pfv_per_hundred'].transform(lambda s: s.rolling(3, min_periods=1).mean())


def plot_by_group_dual_axis(data, group_col, title_prefix, save_path=None):
    """Plot vaccination vs new cases per 100k, grouped by continent or WHO region."""
    d = data.dropna(subset=[group_col]).copy()

    # Aggregate mean values across countries within each group
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

        # 🟠 Vaccination (left axis)
        ax.plot(sub['date'], sub['pfv_per_hundred'], color='orange', label='PfV per 100', linewidth=2)
        ax.set_ylabel('PfV per 100 people', color='orange')
        ax.tick_params(axis='y', labelcolor='orange')
        ax.grid(True)
        ax.set_title(g)

        # 🔵 New Cases (right axis)
        ax2 = ax.twinx()
        ax2.plot(sub['date'], sub['new_cases_per_100k'], color='blue', label='New Cases per 100k', linewidth=2, linestyle='--')
        ax2.set_ylabel('New Cases per 100k', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Combine legends (show once)
        if i == 0:
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

        ax.set_xlabel('Date')

    # Hide any empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(f"{title_prefix}: Vaccination vs New Cases (Dual Axis)", y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.show()


# --- 1️⃣ Plot by Continent ---
plot_by_group_dual_axis(
    df,
    group_col='continent',
    title_prefix='By Continent',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/continent_dual_axis_colored.png"
)

# --- 2️⃣ Plot by WHO Region ---
plot_by_group_dual_axis(
    df,
    group_col='who_region',
    title_prefix='By WHO Region',
    save_path="C:/Users/dipac/Downloads/covid-vax-project/who_dual_axis_colored.png"
)
