import os
import numpy as np
import pandas as pd

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
INPUT = os.path.join(BASE, "clean_weekly_with_100k.csv")

# ===================== RPCA FUNCTIONS =====================

def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def svt(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vt

def robust_pca(M, lam=None, mu=None, max_iter=2000, tol=1e-7, rho=1.5):
    """
    Solve:  min ||L||_* + lam ||S||_1  s.t.  M = L + S
    via IALM. Returns (L, S).
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))   # standard default

    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M, np.inf) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = M / (dual_norm + 1e-12)

    if mu is None:
        mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)

    M_fro = np.linalg.norm(M, 'fro') + 1e-12

    for it in range(max_iter):
        # L-update (nuclear norm via SVT)
        L = svt(M - S + (1.0/mu) * Y, 1.0/mu)

        # S-update (elementwise soft-threshold)
        S = shrink(M - L + (1.0/mu) * Y, lam/mu)

        # residual and dual update
        R = M - L - S
        Y = Y + mu * R

        err = np.linalg.norm(R, 'fro') / M_fro
        if it % 50 == 0:
            print(f"iter {it}: err = {err:.3e}")
        if err < tol:
            print(f"Converged at iter {it}, err = {err:.3e}")
            break

        mu = min(mu * rho, mu_bar)

    return L, S

# ===================== LOAD DATA =====================

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date")

# Ensure numeric
df["new_cases_per_100k"] = pd.to_numeric(df["new_cases_per_100k"], errors="coerce")
df["pfv_per_hundred"]    = pd.to_numeric(df["pfv_per_hundred"],    errors="coerce")
df["population"]         = pd.to_numeric(df["population"],         errors="coerce")

# ===================== WEEKLY AGGREGATION =====================

# week start (Monday by default)
df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

# For each continent + who_region + week:
# - sum new_cases_per_100k over the week
# - take the LAST weekly value of pfv_per_hundred and population
weekly = (
    df.groupby(["continent", "who_region", "week"], as_index=False)
      .agg({
          "new_cases_per_100k": "sum",   # weekly sum of new cases
          "pfv_per_hundred":    "last",  # latest cumulative % that week
          "population":         "last"   # latest population estimate that week
      })
)

weekly_path = os.path.join(BASE, "clean_weekly_SUM_latest.csv")
weekly.to_csv(weekly_path, index=False)
print("Saved weekly aggregated file:", weekly_path)

# ===================== RPCA RUNNER =====================

def run_rpca_and_save(df_weekly, group_col, prefix):
    """
    df_weekly: aggregated weekly dataframe
    group_col: 'continent' or 'who_region'
    prefix: filename prefix
    """
    # Pivot to matrix: rows = weeks, columns = group (e.g., continents)
    mat = (
    df_weekly.pivot_table(
        index="week",
        columns=group_col,
        values="new_cases_per_100k",
        aggfunc="sum"     # sum across WHO regions (or continents) in same week
    )
    .sort_index()
    .fillna(0.0)
)

    

    M = mat.values.astype(float)

    print(f"\n=== Running RPCA for {group_col} ===")
    print("Matrix shape:", M.shape)

    L, S = robust_pca(M)

    # Back to DataFrames
    L_df = pd.DataFrame(L, index=mat.index, columns=mat.columns)
    S_df = pd.DataFrame(S, index=mat.index, columns=mat.columns)

    # Save matrix + components
    mat_path = os.path.join(BASE, f"{prefix}_matrix.csv")
    low_path = os.path.join(BASE, f"{prefix}_lowrank.csv")
    spr_path = os.path.join(BASE, f"{prefix}_sparse.csv")

    mat.to_csv(mat_path)
    L_df.to_csv(low_path)
    S_df.to_csv(spr_path)

    print("Saved:")
    print("  ", mat_path)
    print("  ", low_path)
    print("  ", spr_path)

# ===================== RUN RPCA =====================

# By continent
run_rpca_and_save(weekly, "continent", "rpca_continent_cases")

# By WHO region
run_rpca_and_save(weekly, "who_region", "rpca_who_cases")

print("\n✅ Done: weekly SUM/latest aggregation + RPCA (default λ).")
