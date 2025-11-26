import os
import sys
import subprocess
import pandas as pd
import numpy as np

# ---------- Config ----------
BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
INPUT = os.path.join(BASE, "clean_weekly_with_100k.csv")  # your latest file
SAVE_MATRICES = True

# ---------- Ensure RPCA available ----------
try:
    from r_pca import R_pca
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "r_pca"])
    from r_pca import R_pca

# ---------- Load & prep ----------
df = pd.read_csv(INPUT, encoding="cp1252")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Keep only what we need
assert {"date", "continent", "who_region", "new_cases_per_100k"}.issubset(df.columns), \
    "Dataset must include date, continent, who_region, new_cases_per_100k."

# Fill weird numeric issues
df["new_cases_per_100k"] = pd.to_numeric(df["new_cases_per_100k"], errors="coerce").fillna(0.0)

def rpca_from_pivot(data: pd.DataFrame, group_col: str, value_col: str, name: str):
    """
    Build a date x group matrix, run RPCA, and save M, L, S.
    """
    # Build wide matrix: rows = dates, cols = groups
    mat = (data
           .pivot_table(index="date", columns=group_col, values=value_col, aggfunc="mean")
           .sort_index())

    # Fill missing with 0; RPCA expects a dense numeric matrix.
    mat = mat.fillna(0.0)

    # Save original matrix
    m_path = os.path.join(BASE, f"{name}_matrix.csv")
    if SAVE_MATRICES:
        mat.to_csv(m_path)

    # Run Robust PCA
    M = mat.values.astype(float)
    rpca = R_pca(M)
    L, S = rpca.fit(max_iter=1000, iter_print=100)

    low_rank = pd.DataFrame(L, index=mat.index, columns=mat.columns)
    sparse   = pd.DataFrame(S, index=mat.index, columns=mat.columns)

    # Save results
    L_path = os.path.join(BASE, f"{name}_lowrank.csv")
    S_path = os.path.join(BASE, f"{name}_sparse.csv")
    low_rank.to_csv(L_path)
    sparse.to_csv(S_path)

    print(f"✅ Saved: {m_path}")
    print(f"✅ Saved: {L_path}")
    print(f"✅ Saved: {S_path}")

    return mat, low_rank, sparse

# ---------- Run RPCA: by continent ----------
M_cont, L_cont, S_cont = rpca_from_pivot(
    df, group_col="continent", value_col="new_cases_per_100k", name="rpca_continent_cases_per100k"
)

# ---------- Run RPCA: by WHO region (if present) ----------
if "who_region" in df.columns:
    M_who, L_who, S_who = rpca_from_pivot(
        df, group_col="who_region", value_col="new_cases_per_100k", name="rpca_who_cases_per100k"
    )
