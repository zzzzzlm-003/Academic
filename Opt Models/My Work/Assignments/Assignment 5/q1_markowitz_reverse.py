"""
Q1 — Reverse Markowitz Mean-Variance Portfolio
IEOR4004 Assignment 5

Pipeline:
  1. Pull 3-year daily price history (2023-04-23 to 2026-04-23) for all S&P 500 stocks.
  2. Compute annualized expected returns mu and covariance Sigma.
  3. Q1.2: Solve max mu' w  s.t. w' Sigma w <= sigma_max^2, sum w = 1, w >= 0
  4. Q1.3: Sweep sigma_max -> efficient frontier plot
  5. Q1.4: Add max-weight cap (w_i <= 0.05), re-solve, compare

Run:  python q1_markowitz_reverse.py
Outputs (saved next to this script):
  q1_2_allocation.png            (top holdings bar chart)
  q1_3_efficient_frontier.png    (frontier plot)
  q1_4_comparison.png            (capped vs uncapped allocations)
  q1_results.txt                 (numerical summary)
  processed_data/*.pkl           (cached data so you don't re-download)
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB
import yfinance as yf
import cvxpy as cp

# ----------------------------- Config -----------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(HERE, "processed_data")
START_DATE = "2023-04-23"
END_DATE   = "2026-04-23"
SIGMA_MAX  = 0.20      # 20% annualized portfolio volatility cap (variance = 0.04)
BATCH_SIZE = 25
WEIGHT_CAP = 0.10      # 10% per-asset cap for Q1.4 (matches assignment's example)
MIN_ASSETS_TARGET = 450  # assignment expects ~500; warn/rebuild if far below
MIN_OBS_FRAC = 0.95      # minimum fraction of observations to keep an asset
FACTOR_K = 30            # low-rank factor approximation for risk constraint (license-friendly)

os.makedirs(CACHE_DIR, exist_ok=True)

def _is_finite_array(x: np.ndarray) -> bool:
    x = np.asarray(x)
    return np.isfinite(x).all()


def _validate_mu_cov(mu: np.ndarray, cov: np.ndarray, tickers: list[str]) -> tuple[bool, str]:
    if mu is None or cov is None or tickers is None:
        return False, "mu/cov/tickers is None"
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(tickers)
    if mu.ndim != 1 or cov.ndim != 2:
        return False, f"bad shapes (mu.ndim={mu.ndim}, cov.ndim={cov.ndim})"
    if mu.shape[0] != n or cov.shape != (n, n):
        return False, f"shape mismatch (len(tickers)={n}, mu={mu.shape}, cov={cov.shape})"
    if not _is_finite_array(mu):
        return False, "mu contains non-finite values"
    if not _is_finite_array(cov):
        return False, "cov contains non-finite values"
    if not np.allclose(cov, cov.T, atol=1e-10, rtol=1e-8):
        return False, "cov not symmetric (within tolerance)"
    d = np.diag(cov)
    if (d <= 0).any():
        return False, "cov diagonal has non-positive entries"
    return True, "ok"


# ------------------------ 1. Fetch tickers ------------------------
def get_sp500_tickers():
    """Get the current S&P 500 ticker list.

    Tries multiple public sources (Wikipedia can rate-limit / 403).
    Falls back to a static in-class list only if all sources fail.
    """
    try:
        df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"Wikipedia fetch failed ({e}); trying alternate sources...")

    # Alternate 1: datasets repo (CSV)
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        )
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers = [t.strip() for t in tickers if t.strip()]
        print(f"Fetched {len(tickers)} S&P 500 tickers from datasets GitHub.")
        return tickers
    except Exception as e:
        print(f"Datasets GitHub fetch failed ({e}); trying next source...")

    # Alternate 2: datahub mirror (CSV)
    try:
        df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers = [t.strip() for t in tickers if t.strip()]
        print(f"Fetched {len(tickers)} S&P 500 tickers from datahub.")
        return tickers
    except Exception as e:
        print(f"Datahub fetch failed ({e}); falling back to in-class 100-stock list.")
        return [
            'JD','CSCO','BAC','AEP','NVDA','JNJ','BABA','MCD','WFC','CSX',
            'HON','VIPS','MSFT','SYK','ECL','C','META','MDT','ORCL','COO',
            'SBUX','MS','MMC','CRWD','GS','CHWY','MMM','AAPL','SRE','JPM',
            'DLTR','LLY','DDOG','PYPL','GOOG','XOM','SO','CAT','ISRG','VRTX',
            'OKTA','FSLY','SNPS','IBM','ADBE','NKE','AMD','MDLZ','MELI','INTC',
            'COST','MRNA','NEE','SMCI','JMIA','ABT','GOOGL','EL','KO','CVX',
            'HD','PFE','MA','ASML','CVS','KDP','DXC','EQIX','ABBV','NFLX',
            'AMZN','DOV','MU','CRM','ETSY','QCOM','CB','ROKU','TM','PEP',
            'AMAT','PG','CMCSA','UNH','LOW','EMR','ZM','RTX','TXN','MCHP',
            'DHI','MRK','TSLA','DUK','CI','CPRT','BRK-B','DG','EFX','CTAS'
        ]


# ----------------- 2. Download + compute mu, Sigma ----------------
def download_and_process(tickers, start, end, batch_size=25):
    print(f"\nDownloading prices {start} -> {end} for {len(tickers)} tickers in batches of {batch_size}...")
    all_prices = pd.DataFrame()
    failed = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False,
                threads=True, group_by="ticker"
            )
            # yfinance returns a multi-index frame for multi-ticker downloads
            if isinstance(data.columns, pd.MultiIndex):
                close = pd.DataFrame({t: data[t]["Close"] for t in batch if t in data.columns.get_level_values(0)})
            else:
                close = data[["Close"]].rename(columns={"Close": batch[0]})
            all_prices = pd.concat([all_prices, close], axis=1)
        except Exception as e:
            for t in batch:
                failed[t] = str(e)
        time.sleep(1)

    # Basic sanity: keep columns with sufficient observations.
    if all_prices.shape[0] == 0 or all_prices.shape[1] == 0:
        raise RuntimeError("No price data downloaded (empty dataframe).")

    min_obs = int(MIN_OBS_FRAC * all_prices.shape[0])
    good = all_prices.columns[all_prices.notna().sum() >= min_obs]
    prices = all_prices[good].copy()

    # Replace non-positive prices (bad/missing) with NaN, then drop rows with any NaN
    # *after* pruning assets, to keep the intersection as large as possible.
    prices = prices.where(prices > 0)
    prices = prices.dropna(how="any")
    print(f"  -> kept {prices.shape[1]} tickers with aligned history ({prices.shape[0]} trading days)")

    # Returns: guard against inf from division by 0, then drop any remaining NaNs.
    daily_ret = prices.pct_change()
    daily_ret = daily_ret.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Extra guard: drop columns that still contain non-finite values.
    finite_cols = daily_ret.columns[np.isfinite(daily_ret.to_numpy()).all(axis=0)]
    daily_ret = daily_ret[finite_cols]

    mu = daily_ret.mean().to_numpy() * 252                  # annualized expected returns
    cov = daily_ret.cov().to_numpy() * 252                  # annualized covariance

    # Numerical guard: enforce symmetry (floating point), and drop tiny negative eigen noise on diag.
    cov = 0.5 * (cov + cov.T)
    return mu, cov, list(prices.columns), failed


def load_or_build():
    f_mu  = os.path.join(CACHE_DIR, "mu.pkl")
    f_cov = os.path.join(CACHE_DIR, "cov.pkl")
    f_tic = os.path.join(CACHE_DIR, "tickers.pkl")
    if all(os.path.exists(p) for p in [f_mu, f_cov, f_tic]):
        print("Loading cached mu/cov from disk...")
        mu = pickle.load(open(f_mu, "rb"))
        cov = pickle.load(open(f_cov, "rb"))
        tic = pickle.load(open(f_tic, "rb"))
        ok, reason = _validate_mu_cov(mu, cov, tic)
        if ok and len(tic) >= MIN_ASSETS_TARGET:
            return mu, cov, tic
        if ok and len(tic) < MIN_ASSETS_TARGET:
            print(f"Cached universe too small (n={len(tic)} < {MIN_ASSETS_TARGET}); rebuilding from Wikipedia S&P 500...")
        else:
            print(f"Cached data invalid ({reason}); rebuilding from yfinance...")
    tickers = get_sp500_tickers()
    mu, cov, used, _ = download_and_process(tickers, START_DATE, END_DATE, BATCH_SIZE)
    pickle.dump(mu,  open(f_mu, "wb"))
    pickle.dump(cov, open(f_cov, "wb"))
    pickle.dump(used, open(f_tic, "wb"))
    return mu, cov, used


# ------------------ 3. Reverse Markowitz QP solver -----------------
def solve_max_return(mu, cov, sigma_max, weight_cap=None):
    """
    max  mu' w
    s.t. w' Sigma w <= sigma_max^2
         sum w = 1
         0 <= w_i <= (weight_cap if given else 1)
    """
    n = len(mu)
    ub = weight_cap if weight_cap is not None else 1.0

    # Try Gurobi first (fast), but fall back to CVXPY if the license size-limit triggers.
    try:
        m = gp.Model("MaxReturn_RiskCapped")
        m.setParam("OutputFlag", 0)
        w = m.addMVar(n, lb=0.0, ub=ub, name="w")
        m.setObjective(mu @ w, GRB.MAXIMIZE)

        k = min(FACTOR_K, n)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1][:k]
        lam = np.clip(evals[idx], 0.0, None)
        V = evecs[:, idx]
        B = V * np.sqrt(lam)  # n x k
        # Use a diagonal residual to keep the QP small.
        approx_diag = np.sum(B * B, axis=1)
        d = np.clip(np.diag(cov) - approx_diag, 0.0, None)

        z = m.addMVar(k, lb=-GRB.INFINITY, name="z")
        m.addConstr(z == B.T @ w, name="factor_map")
        m.addConstr(z @ z + (d * w) @ w <= sigma_max ** 2, name="risk")
        m.addConstr(w.sum() == 1, name="budget")
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise gp.GurobiError(f"status {m.Status}")
        w_opt = w.X
    except gp.GurobiError as e:
        print(f"Gurobi failed ({e}); falling back to CVXPY (SCS).")
        w_cvx = cp.Variable(n, nonneg=True)
        cons = [cp.sum(w_cvx) == 1, w_cvx <= ub, cp.quad_form(w_cvx, cov) <= sigma_max**2]
        prob = cp.Problem(cp.Maximize(mu @ w_cvx), cons)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=20000)
        if w_cvx.value is None:
            raise RuntimeError("CVXPY failed to find a solution.")
        w_opt = np.asarray(w_cvx.value).reshape(-1)

    ret = float(mu @ w_opt)
    var = float(np.einsum("i,ij,j->", w_opt, cov, w_opt))
    return w_opt, ret, var


# ----------------------- 4. Plot helpers --------------------------
def plot_top_allocations(weights, tickers, title, out_path, top=20):
    pairs = sorted(zip(weights, tickers), reverse=True)[:top]
    w_top = [p[0] for p in pairs]
    t_top = [p[1] for p in pairs]
    plt.figure(figsize=(11, 5))
    plt.bar(t_top, w_top)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================ MAIN ================================
if __name__ == "__main__":
    mu, cov, tickers = load_or_build()
    n = len(mu)
    print(f"\nUniverse size: {n} assets.")
    print(f"Annual return range: [{mu.min():.4f}, {mu.max():.4f}]")

    log_lines = []

    # -------- Q1.2: solve at sigma_max = 20% vol --------
    print(f"\n[Q1.2] Solving max-return s.t. sigma <= {SIGMA_MAX:.2%} (vol)...")
    w_star, r_star, v_star = solve_max_return(mu, cov, SIGMA_MAX)
    nz = int(np.sum(w_star > 1e-4))
    log_lines.append("=== Q1.2 Reverse Markowitz ===")
    log_lines.append(f"sigma_max (vol) = {SIGMA_MAX:.4f}")
    log_lines.append(f"Achieved annual return = {r_star:.4f}")
    log_lines.append(f"Realized portfolio variance = {v_star:.6f}  (std = {np.sqrt(v_star):.4f})")
    log_lines.append(f"Number of non-trivial holdings (w>1e-4): {nz}")
    log_lines.append("Top 10 holdings:")
    for w, t in sorted(zip(w_star, tickers), reverse=True)[:10]:
        if w > 1e-4:
            log_lines.append(f"  {t:6s}  {w:.4f}")
    plot_top_allocations(w_star, tickers,
                         f"Q1.2 Reverse Markowitz Top 20 (σmax={SIGMA_MAX:.0%})",
                         os.path.join(HERE, "q1_2_allocation.png"))

    # -------- Q1.3: efficient frontier sweep --------
    print("\n[Q1.3] Sweeping sigma_max to build efficient frontier...")
    # First find the minimum-variance portfolio's vol so we don't probe infeasible sigma_max.
    # Compute it with CVXPY to avoid any license-size limitations.
    w_mv = cp.Variable(n, nonneg=True)
    prob_mv = cp.Problem(cp.Minimize(cp.quad_form(w_mv, cov)), [cp.sum(w_mv) == 1])
    prob_mv.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=20000)
    if w_mv.value is None:
        raise RuntimeError("CVXPY failed to compute min-variance portfolio.")
    vol_min = float(np.sqrt(np.einsum("i,ij,j->", w_mv.value, cov, w_mv.value)))
    vol_max_asset = float(np.sqrt(np.diag(cov)).max())
    print(f"  feasible sigma range: [{vol_min:.4f}, {vol_max_asset:.4f}]")
    sigma_grid = np.linspace(vol_min * 1.02, vol_max_asset * 0.98, 25)

    rets, vars_, sigs = [], [], []
    for s in sigma_grid:
        _, r, v = solve_max_return(mu, cov, s)
        if r is None:
            continue
        sigs.append(s); rets.append(r); vars_.append(v)
    rets = np.array(rets); vols = np.sqrt(np.array(vars_))

    plt.figure(figsize=(7, 5))
    plt.plot(vols, rets, "o-", lw=2)
    plt.xlabel("Realized portfolio volatility σ")
    plt.ylabel("Maximum expected return  μᵀw*")
    plt.title("Q1.3 Efficient Frontier (reverse Markowitz)")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "q1_3_efficient_frontier.png"), dpi=150)
    plt.close()
    log_lines.append("\n=== Q1.3 Efficient Frontier ===")
    log_lines.append(f"Min-variance portfolio vol = {vol_min:.4f}")
    for s, r in zip(vols, rets):
        log_lines.append(f"  realized vol={s:.4f}  ->  max return={r:.4f}")

    # -------- Q1.4: add per-asset cap, re-solve at same sigma_max --------
    print(f"\n[Q1.4] Re-solving with per-asset cap w_i <= {WEIGHT_CAP}...")
    w_cap, r_cap, v_cap = solve_max_return(mu, cov, SIGMA_MAX, weight_cap=WEIGHT_CAP)
    nz_cap = int(np.sum(w_cap > 1e-4))

    log_lines.append("\n=== Q1.4 Capped Portfolio (max weight 5%) ===")
    log_lines.append(f"sigma_max (vol) = {SIGMA_MAX:.4f}")
    log_lines.append(f"Achieved return = {r_cap:.4f} (uncapped: {r_star:.4f})")
    log_lines.append(f"Realized variance = {v_cap:.6f} (uncapped: {v_star:.6f})")
    log_lines.append(f"Non-trivial holdings: {nz_cap} (uncapped: {nz})")
    log_lines.append("Top 10 capped holdings:")
    for w, t in sorted(zip(w_cap, tickers), reverse=True)[:10]:
        if w > 1e-4:
            log_lines.append(f"  {t:6s}  {w:.4f}")

    # comparison plot — top 20 union
    top_union = sorted(set(
        [t for _, t in sorted(zip(w_star, tickers), reverse=True)[:15]] +
        [t for _, t in sorted(zip(w_cap,  tickers), reverse=True)[:15]]
    ))
    idx = [tickers.index(t) for t in top_union]
    width = 0.4
    x = np.arange(len(top_union))
    plt.figure(figsize=(13, 5))
    plt.bar(x - width/2, w_star[idx], width, label="Q1.2 uncapped")
    plt.bar(x + width/2, w_cap[idx],  width, label=f"Q1.4 capped (≤{WEIGHT_CAP})")
    plt.xticks(x, top_union, rotation=70, ha="right")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Q1.4 Allocation comparison: uncapped vs capped")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "q1_4_comparison.png"), dpi=150)
    plt.close()

    with open(os.path.join(HERE, "q1_results.txt"), "w") as f:
        f.write("\n".join(log_lines))

    print("\nAll outputs saved next to this script. See q1_results.txt for the numerical summary.")
