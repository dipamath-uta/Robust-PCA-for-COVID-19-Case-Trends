import os
import numpy as np
import pandas as pd

# ----------------- Paths -----------------
BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
INPUT = os.path.join(BASE, "clean_weekly_with_100k.csv")

# ----------------- Robust PCA (IALM) -----------------
def shrink(X, tau):
    # elementwise soft-threshold
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def svt(X, tau):
    # singular value thresholding
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vt

def robust_pca(M, lam=None, mu=None, max_iter=1000, tol=1e-7, rho=1.5):
    """
    Solve:  min ||L||_* + lam ||S||_1  s.t.  M = L + S
    IALM algorithm. Returns (L, S).
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))   # common default
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M, np.inf) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = M / dual_norm

    if mu is None:
        mu = 1.25 / norm_two if norm_two > 0 else 1.25
    mu_bar = mu * 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)

    M_fro = np.linalg.norm(M, 'fro') + 1e-12

    for _ in range(max_iter):
        # L-update: SVT
        L = svt(M - S + (1.0/mu) * Y, 1.0/mu)

        # S-update: shrinkage
        S = shrink(M - L + (1.0/mu) * Y, lam/mu)

        # dual update
        R = M - L - S                      # residual
        Y = Y + mu * R

        # convergence check
        err = np.linalg.norm(R, 'fro') / M_fro
        if err < tol:
            break

        # step update
        mu = min(mu * rho, mu_bar)

    return L, S

# ----------------- Load & prep -----------------
df = pd.read_csv(INPUT, encoding="cp1252")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ensure numeric
df["new_cases_per_100k"] = pd.to_numeric(df["new_cases_per_100k"], errors="coerce").fillna(0.0)

def run_rpca_and_save(data: pd.DataFrame, group_col: str, prefix: str):
    # pivot to date × group matrix
    mat = (data.pivot_table(index="date", columns=group_col,
                            values="new_cases_per_100k", aggfunc="mean")
                .sort_index()
                .fillna(0.0))
    M = mat.values.astype(float)

    # run RPCA
    L, S = robust_pca(M, max_iter=2000, tol=1e-6)

    # back to DataFrames
    L_df = pd.DataFrame(L, index=mat.index, columns=mat.columns)
    S_df = pd.DataFrame(S, index=mat.index, columns=mat.columns)

    # save all three
    m_path = os.path.join(BASE, f"{prefix}_matrix.csv")
    l_path = os.path.join(BASE, f"{prefix}_lowrank.csv")
    s_path = os.path.join(BASE, f"{prefix}_sparse.csv")
    mat.to_csv(m_path)
    L_df.to_csv(l_path)
    S_df.to_csv(s_path)
    print(f"Saved:\n  {m_path}\n  {l_path}\n  {s_path}")

# ----------------- By Continent -----------------
run_rpca_and_save(df, group_col="continent", prefix="rpca_continent_cases_per100k")

# ----------------- By WHO Region -----------------
if "who_region" in df.columns:
    run_rpca_and_save(df, group_col="who_region", prefix="rpca_who_cases_per100k")
else:
    print("Column 'who_region' not found; skipped WHO RPCA.")
