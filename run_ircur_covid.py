import numpy as np
import pandas as pd

# import IRCUR from the cloned GitHub repo
# (make sure the repo folder containing robustpca/ is on your PYTHONPATH,
#  or that this script lives in the same project and you use a local package install)
from robustpca.ircur import IRCUR


def run_ircur_on_matrix(
    csv_path: str,
    out_lowrank_path: str,
    out_sparse_path: str,
    rank: int = 2,
    nr: int = 200,
    nc: int = 6,
    tol: float = 1e-5,
    thresholding_decay: float = 0.65,
    resample: bool = True,
    max_iter: int = 1e4,
    verbose: bool = True,
) -> None:
    """
    Load a date×region matrix from csv_path, run IRCUR, and
    save low-rank and sparse components to CSV.
    """
    # --- load data ---
    df = pd.read_csv(csv_path, index_col=0)
    # ensure numeric and drop rows with NaNs if any
    M = df.astype(float).values
    n, m = M.shape

    # clip nr/nc to matrix size
    nr = min(nr, n)
    nc = min(nc, m)

    # pick a reasonable initial threshold scale
    # (you can tune this; 0.5 * max abs value is a decent starting point)
    initial_threshold = 0.5 * np.max(np.abs(M))

    ircur = IRCUR()
    L, S = ircur.decompose(
        data_mat=M,
        rank=rank,
        nr=nr,
        nc=nc,
        initial_threshold=initial_threshold,
        tol=tol,
        thresholding_decay=thresholding_decay,
        resample=resample,
        max_iter=max_iter,
        verbose=verbose,
    )

    # --- save with same index/columns as input ---
    L_df = pd.DataFrame(L, index=df.index, columns=df.columns)
    S_df = pd.DataFrame(S, index=df.index, columns=df.columns)

    L_df.to_csv(out_lowrank_path)
    S_df.to_csv(out_sparse_path)

    if verbose:
        print(f"Saved IRCUR low-rank to   {out_lowrank_path}")
        print(f"Saved IRCUR sparse to     {out_sparse_path}")


if __name__ == "__main__":
    # === EXAMPLES: adjust these file names to your actual ones ===

    # Continents – weekly mean cases per 100k
    run_ircur_on_matrix(
        csv_path="rpca_continent_cases_matrix.csv",
        out_lowrank_path="weekly_continent_cases_per100k_ircur_lowrank.csv",
        out_sparse_path="weekly_continent_cases_per100k_ircur_sparse.csv",
        rank=2,      # or 3, depending on how many global “waves” you want
        nr=200,
        nc=6,
    )

    # WHO regions – weekly mean cases per 100k
    run_ircur_on_matrix(
        csv_path="rpca_who_cases_matrix.csv",
        out_lowrank_path="weekly_who_cases_per100k_ircur_lowrank.csv",
        out_sparse_path="weekly_who_cases_per100k_ircur_sparse.csv",
        rank=2,
        nr=200,
        nc=6,
    )
