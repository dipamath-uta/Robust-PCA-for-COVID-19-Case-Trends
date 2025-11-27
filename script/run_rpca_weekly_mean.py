import os
import numpy as np
import pandas as pd

BASE = r"C:/Users/dipac/Downloads/covid-vax-project"
INPUT = os.path.join(BASE, "clean_weekly_MEAN_latest.csv")

# ============================================================
# 1) PCP (convex RPCA) utilities
# ============================================================

def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def svt(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vt

def pcp(M, lam=None, mu=None, max_iter=1000, tol=1e-7, rho=1.5, verbose=False):
    """
    Classic convex RPCA (PCP) via IALM:
        min ||L||_* + lam ||S||_1  s.t. M = L + S
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M, np.inf) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = M / (dual_norm + 1e-12)

    if mu is None:
        mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)

    M_fro = np.linalg.norm(M, "fro") + 1e-12

    for k in range(max_iter):
        # L update
        L = svt(M - S + (1.0 / mu) * Y, 1.0 / mu)
        # S update
        S = shrink(M - L + (1.0 / mu) * Y, lam / mu)

        R = M - L - S
        Y = Y + mu * R

        err = np.linalg.norm(R, "fro") / M_fro
        if verbose and k % 50 == 0:
            print(f"[PCP] iter {k}, err={err:.3e}")
        if err < tol:
            if verbose:
                print(f"[PCP] converged at iter {k}, err={err:.3e}")
            break

        mu = min(mu * rho, mu_bar)

    return L, S

# ============================================================
# 2) IRLS-RPCA (working version from irls_rpca_latest.py)
# ============================================================

def weighted_svt(X, base_tau, s_prev=None, eps=1e-6):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s_prev is None:
        tau_vec = np.full_like(s, base_tau)
    else:
        w = 1.0 / (np.abs(s_prev) + eps)
        w = w / w.mean()
        tau_vec = base_tau * w
    s_new = np.maximum(s - tau_vec, 0.0)
    return (U * s_new) @ Vt, s_new

def irls_rpca(M, lam=None, max_iter=1000, tol=1e-7, rho=1.5, verbose=False):
    """
    IRLS-style non-convex RPCA:
        M = L + S
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M, np.inf) / lam
    dual_norm = max(norm_two, norm_inf)
    Y = M / (dual_norm + 1e-12)

    mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    M_fro = np.linalg.norm(M, "fro") + 1e-12

    s_prev = None

    for k in range(max_iter):
        # L update: weighted SVT
        X_L = M - S + (1.0 / mu) * Y
        L, s_prev = weighted_svt(X_L, base_tau=1.0 / mu, s_prev=s_prev)

        # S update: soft-threshold
        X_S = M - L + (1.0 / mu) * Y
        S = shrink(X_S, lam / mu)

        # dual update
        R = M - L - S
        Y = Y + mu * R

        err = np.linalg.norm(R, "fro") / M_fro
        if verbose and k % 50 == 0:
            print(f"[IRLS] iter {k}, err={err:.3e}")
        if err < tol:
            if verbose:
                print(f"[IRLS] converged at iter {k}, err={err:.3e}")
            break

        mu = min(mu * rho, mu_bar)

    return L, S

# ============================================================
# 3) Run on continent & WHO matrices (weekly MEAN)
# ============================================================

df = pd.read_csv(INPUT, parse_dates=["week"]).sort_values("week")

def build_matrix(df_src, group_col):
    mat = (
        df_src.pivot_table(index="week",
                           columns=group_col,
                           values="new_cases_per_100k",
                           aggfunc="mean")
            .sort_index()
            .fillna(0.0)
    )
    return mat

def run_all_for_group(df_src, group_col,
                      matrix_name,
                      pcp_low_name, pcp_sparse_name,
                      irls_low_name, irls_sparse_name):
    mat = build_matrix(df_src, group_col)
    M = mat.values.astype(float)

    # Save matrix
    matrix_path = os.path.join(BASE, matrix_name)
    mat.to_csv(matrix_path)
    print("Saved matrix:", matrix_path)

    # Convex PCP
    print(f"\n=== PCP on {group_col} matrix ===")
    L_pcp, S_pcp = pcp(M, verbose=True)
    pd.DataFrame(L_pcp, index=mat.index, columns=mat.columns).to_csv(
        os.path.join(BASE, pcp_low_name)
    )
    pd.DataFrame(S_pcp, index=mat.index, columns=mat.columns).to_csv(
        os.path.join(BASE, pcp_sparse_name)
    )
    print("  -> PCP low-rank:", pcp_low_name)
    print("  -> PCP sparse  :", pcp_sparse_name)

    # IRLS RPCA
    print(f"\n=== IRLS-RPCA on {group_col} matrix ===")
    L_i, S_i = irls_rpca(M, verbose=True)
    pd.DataFrame(L_i, index=mat.index, columns=mat.columns).to_csv(
        os.path.join(BASE, irls_low_name)
    )
    pd.DataFrame(S_i, index=mat.index, columns=mat.columns).to_csv(
        os.path.join(BASE, irls_sparse_name)
    )
    print("  -> IRLS low-rank:", irls_low_name)
    print("  -> IRLS sparse  :", irls_sparse_name)


# ---------- Continents ----------
run_all_for_group(
    df, "continent",
    matrix_name      = "rpca_continent_cases_matrix.csv",
    pcp_low_name     = "rpca_continent_cases_lowrank.csv",
    pcp_sparse_name  = "rpca_continent_cases_sparse.csv",
    irls_low_name    = "continent_cases_per100k_irls_lowrank.csv",
    irls_sparse_name = "continent_cases_per100k_irls_sparse.csv"
)

# ---------- WHO regions ----------
run_all_for_group(
    df, "who_region",
    matrix_name      = "rpca_who_cases_matrix.csv",
    pcp_low_name     = "rpca_who_cases_lowrank.csv",
    pcp_sparse_name  = "rpca_who_cases_sparse.csv",
    irls_low_name    = "who_cases_per100k_irls_lowrank.csv",
    irls_sparse_name = "who_cases_per100k_irls_sparse.csv"
)

print("\n✅ Weekly MEAN RPCA (PCP + IRLS) finished.")
