#!/usr/bin/env python3
"""
GRAVITY ARENA
Python translation of the Pine Script v6 indicator.

Scores 40 crypto assets on 7 factors (Omega, ROC, Sortino, Z-Score, RSI, Alpha, ADX),
selects top-N longs (score==7) ranked by alpha descending,
then builds Top-1 and Top-4 equal-weight equity curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from utils.backtest import Backtester

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
BACKTEST_START = "2023-01-01"
TIMEFRAME = "8h"
FEE_PER_TRADE = 0.0005  # 0.05% per side

# Data range (includes warmup before BACKTEST_START)
start_date = "2023-01-01"
end_date = str(datetime.now().date())


ROC_LENGTH = 50
OMEGA_LENGTH = 35
SORTINO_LENGTH = 45
ZSCORE_LENGTH = 20
RSI_LENGTH = 50
RSI_THRESHOLD = 55
BETA_LEN = 30
ALPHA_LEN = 17
ADX_SMOOTHING = 15
DI_LENGTH = 15
ADX_SMA_LEN = 20
ROC_THRESHOLD = 0.0
ZSCORE_MEAN = 1.0


# ASSETS = {
#     "ETH":   "BNB/USDT",
#     "BNB":   "XRP/USDT",
#     "XRP":   "SOL/USDT",
#     "SOL":   "DOGE/USDT",
#     "TRX":   "ADA/USDT",
#     "DOGE":  "LTC/USDT",
#     "ADA":   "TRX/USDT",
#     "BCH":   "DOT/USDT",
#     "HYPE":  "SHIB/USDT",
#     "XMR":   "UNI/USDT",
#     "LINK":  "SOL/USDT",
#     "XLM":   "AVAX/USDT",
#     "ZEC":   "LINK/USDT",
#     "LTC":   "TON/USDT",
#     "SUI":   "XMR/USDT",
#     "AVAX":  "ATOM/USDT",
#     "SHIB":  "ETC/USDT",
#     "HBAR":  "XLM/USDT",
#     "TON":   "BCH/USDT",
#     "ENA":   "APE/USDT",
#     "DOT":   "QNT/USDT",
#     "UNI":   "ALGO/USDT",
#     "MNT":   "VET/USDT",
#     "POL":   "ICP/USDT",
#     "TAO":   "FIL/USDT",
#     "AAVE":  "NEAR/USDT",
#     "APT":   "HBAR/USDT",
#     "PEPE":  "EOS/USDT",
#     "NEAR":  "EGLD/USDT",
#     "ICP":   "HT/USDT",
#     "ETC":   "THETA/USDT",
#     "ONDO":  "AAVE/USDT",
#     "KAS":   "BIT/USDT",
#     "ATOM":  "FLOW/USDT",
#     "WLD":   "CHZ/USDT",
#     "BTC":   "BTC/USDT",    # benchmark (asset40)
# }

# ASSETS = {
#     "ETH":   "SOL/USDT",
#     "BNB":   "XRP/USDT",
#     "XRP":   "ADA/USDT",
#     "SOL":   "AVAX/USDT",
#     "TRX":   "DOGE/USDT",
#     "DOGE":  "DOT/USDT",
#     "ADA":   "LINK/USDT",
#     "BCH":   "TON/USDT",
#     "HYPE":  "SHIB/USDT",
#     "XMR":   "ICP/USDT",
#     "LINK":  "LTC/USDT",
#     "CC":    "BCH/USDT",
#     "XLM":   "UNI/USDT",
#     "ZEC":   "ATOM/USDT",
#     "LTC":   "NEAR/USDT",
#     "SUI":   "XLM/USDT",
#     "AVAX":  "FIL/USDT",
#     "SHIB":  "OP/USDT",
#     "HBAR":  "INJ/USDT",
#     "WLFI":  "ETC/USDT",
#     "TON":   "XMR/USDT",
#     "ENA":   "HBAR/USDT",
#     "DOT":   "APT/USDT",
#     "UNI":   "VET/USDT",
#     "MNT":   "LDO/USDT",
#     "POL":   "KAS/USDT",
#     "TAO":   "STX/USDT",
#     "AAVE":  "ARB/USDT",
#     "APT":   "TIA/USDT",
#     "PEPE":  "GRT/USDT",
#     "NEAR":  "ALGO/USDT",
#     "ICP":   "EGLD/USDT",
#     "ETC":   "RUNE/USDT",
#     "ASTER": "RNDR/USDT",
#     "ONDO":  "AAVE/USDT",
#     "SKY":   "QNT/USDT",
#     "KAS":   "SEI/USDT",
#     "ATOM":  "ORDI/USDT",
#     "WLD":   "MINA/USDT",
#     "BTC":   "BTC/USDT",    # benchmark (asset40)
# }
# 
#     
# }

ASSETS = {
    "XLM":      "XLM/USDT",
    "XRP":      "XRP/USDT",
    "HBAR":     "HBAR/USDT",
    "BCH":      "BCH/USDT",
    "AVAX":     "AVAX/USDT",
    "DOGE":     "DOGE/USDT",
    "SHIB":     "SHIB/USDT",
    "CRV":      "CRV/USDT",
    "TON":      "TON/USDT",
    "UNI":      "UNI/USDT",
    "DOT":      "DOT/USDT",
    "JUP":      "JUP/USDT",
    "PEPE":     "PEPE/USDT",
    "AAVE":     "AAVE/USDT",
    "FARTCOIN": "FARTCOIN/USDT",
    "TAO":      "TAO/USDT",
    "ENA":      "ENA/USDT",
    "NEAR":     "NEAR/USDT",
    "ETC":      "ETC/USDT",
    "TIA":      "TIA/USDT",
    "ONDO":     "ONDO/USDT",
    "APT":      "APT/USDT",
    "FLOKI":    "FLOKI/USDT",
    "BONK":     "BONK/USDT",
    "PENGU":    "PENGU/USDT",
    "ZEC":      "ZEC/USDT",
    "MNT":      "MNT/USDT",
    "POL":      "POL/USDT",
    "ALGO":     "ALGO/USDT",
    "SUI":      "SUI/USDT",
    "WIF":      "WIF/USDT",
    "RENDER":   "RENDER/USDT",
    "WLD":      "WLD/USDT",
    "VIRTUAL":  "VIRTUAL/USDT",
    "SEI":      "SEI/USDT",
    "SPX":      "SPX/USDT",
    "ATOM":     "ATOM/USDT",
    "FIL":      "FIL/USDT",
    "FET":      "FET/USDT",
    "BTC":      "BTC/USDT",    # benchmark (asset40)
}

ASSET_NAMES = list(ASSETS.keys())
BENCHMARK_NAME = "BTC"


# ─────────────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS (matching Pine Script logic)
# All use [1] offset convention: indicators computed on confirmed bars
# ─────────────────────────────────────────────────────────────────────

def calc_roc(close, length):
    """Rate of change (as fraction, not percent) using close[1] vs close[1+length]."""
    shifted = close.shift(1)
    roc = shifted / shifted.shift(length) - 1
    return roc


def calc_omega_ratio(close, length, threshold=0.0):
    """Rolling Omega ratio on close[1]/close[2]-1 returns."""
    returns = close.shift(1) / close.shift(2) - 1
    omega = pd.Series(np.nan, index=close.index)

    pos_excess = []
    neg_excess = []

    for i in range(len(returns)):
        r = returns.iloc[i]
        if pd.isna(r):
            continue
        excess = r - threshold
        if excess > 0:
            pos_excess.append(excess)
        elif excess < 0:
            neg_excess.append(abs(excess))

        if len(pos_excess) > length:
            pos_excess.pop(0)
        if len(neg_excess) > length:
            neg_excess.pop(0)

        pos_sum = sum(pos_excess) if pos_excess else 0.0
        neg_sum = sum(neg_excess) if neg_excess else 0.0
        omega.iloc[i] = pos_sum / neg_sum if neg_sum > 0 else np.nan

    return omega


def calc_sortino_ratio(close, length, risk_free_rate=0.0):
    """Rolling Sortino (actually Sharpe using total stdev, matching Pine code)."""
    returns = close.shift(1) / close.shift(2) - 1
    sortino = pd.Series(np.nan, index=close.index)

    returns_buf = []

    for i in range(len(returns)):
        r = returns.iloc[i]
        if pd.isna(r):
            continue
        returns_buf.append(r)
        if len(returns_buf) > length:
            returns_buf.pop(0)

        if len(returns_buf) > 1:
            mean_r = np.mean(returns_buf)
            std_r = np.std(returns_buf, ddof=1)
            if std_r > 0:
                sortino.iloc[i] = (mean_r - risk_free_rate) / std_r * np.sqrt(252)

    return sortino


def calc_zscore(close, length):
    """Z-score of close[1] relative to SMA and stdev of close[1]."""
    shifted = close.shift(1)
    mean = shifted.rolling(length).mean()
    std = shifted.rolling(length).std(ddof=0)  # Pine ta.stdev uses population stdev
    zscore = (shifted - mean) / std
    zscore = zscore.where(std != 0, 0)
    return zscore


def calc_rsi(close, length):
    """RSI of close[1]. SMA seed then Wilder's smoothing, matching Pine ta.rsi."""
    shifted = close.shift(1)
    delta = shifted.diff()

    # Keep gain/loss as 0.0 for negative/positive deltas, but NaN where delta is NaN
    gain = np.where(delta > 0, delta, 0.0)   # NaN > 0 → False → 0.0
    loss = np.where(delta < 0, -delta, 0.0)  # NaN < 0 → False → 0.0

    # Find first valid delta (positions 0 and 1 are NaN due to shift(1)+diff())
    valid_positions = np.where(delta.notna().values)[0]
    if len(valid_positions) < length:
        return pd.Series(np.nan, index=close.index)

    first_valid = valid_positions[0]   # = 2 for a full series
    seed_end = first_valid + length

    if seed_end > len(close):
        return pd.Series(np.nan, index=close.index)

    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)

    # SMA seed over the first `length` valid bars
    avg_gain[seed_end - 1] = gain[first_valid:seed_end].mean()
    avg_loss[seed_end - 1] = loss[first_valid:seed_end].mean()

    # Wilder's smoothing
    alpha = 1.0 / length
    one_minus = 1.0 - alpha
    for j in range(seed_end, len(close)):
        avg_gain[j] = avg_gain[j-1] * one_minus + gain[j] * alpha
        avg_loss[j] = avg_loss[j-1] * one_minus + loss[j] * alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index)


def calc_beta(benchmark_close, asset_close, length):
    """Log-return beta of asset vs benchmark."""
    asset_ret = np.log(asset_close.shift(1) / asset_close.shift(2))
    bench_ret = np.log(benchmark_close.shift(1) / benchmark_close.shift(2))
    avg_a = asset_ret.rolling(length).mean()
    avg_b = bench_ret.rolling(length).mean()
    cov = ((asset_ret - avg_a) * (bench_ret - avg_b)).rolling(length).mean()
    var_b = bench_ret.rolling(length).var(ddof=0)  # Pine ta.variance uses population variance
    beta = cov / var_b
    return beta


def calc_alpha(benchmark_close, asset_close, beta_len, alpha_len):
    """
    Alpha = avg_asset_return - beta * avg_benchmark_return
    Uses averaged beta and returns across 3 windows (matching Pine).
    """
    asset_ret = np.log(asset_close.shift(1) / asset_close.shift(2))
    bench_ret = np.log(benchmark_close.shift(1) / benchmark_close.shift(2))

    beta1 = calc_beta(benchmark_close, asset_close, beta_len)
    beta2 = calc_beta(benchmark_close, asset_close, beta_len + 9)
    beta3 = calc_beta(benchmark_close, asset_close, max(beta_len - 9, 2))
    beta_avg = (beta1 + beta2 + beta3) / 3

    avg_a1 = asset_ret.rolling(alpha_len).mean()
    avg_a2 = asset_ret.rolling(alpha_len + 3).mean()
    avg_a3 = asset_ret.rolling(max(alpha_len - 3, 2)).mean()
    avg_a = (avg_a1 + avg_a2 + avg_a3) / 3

    avg_b1 = bench_ret.rolling(alpha_len).mean()
    avg_b2 = bench_ret.rolling(alpha_len + 3).mean()
    avg_b3 = bench_ret.rolling(max(alpha_len - 3, 2)).mean()
    avg_b = (avg_b1 + avg_b2 + avg_b3) / 3

    expected = beta_avg * avg_b
    alpha = avg_a - expected
    return alpha


def _pine_rma(arr, length):
    """
    Pine ta.rma: SMA seed over first `length` non-NaN values, then Wilder's smoothing.
    Accepts and returns a numpy array.
    """
    result = np.full(len(arr), np.nan)
    valid_pos = np.where(~np.isnan(arr))[0]
    if len(valid_pos) < length:
        return result
    first = valid_pos[0]
    seed_end = first + length
    if seed_end > len(arr):
        return result
    result[seed_end - 1] = arr[first:seed_end].mean()
    alpha = 1.0 / length
    one_minus = 1.0 - alpha
    for j in range(seed_end, len(arr)):
        result[j] = result[j-1] * one_minus + arr[j] * alpha
    return result


def calc_adx(high, low, close, dilen=14, adxlen=14):
    """Calculate ADX, matching Pine's dirmov/adx (ta.rma + fixnan)."""
    h, l, c = high.values, low.values, close.values
    prev_c = np.roll(c, 1); prev_c[0] = np.nan

    up   = np.diff(h, prepend=np.nan)
    down = -np.diff(l, prepend=np.nan)

    # plusDM / minusDM: NaN on bar 0 (matching Pine na(up) ? na : ...)
    plus_dm  = np.where(~np.isnan(up),   np.where((up > down)   & (up > 0),   up,   0.0), np.nan)
    minus_dm = np.where(~np.isnan(down), np.where((down > up)   & (down > 0), down, 0.0), np.nan)

    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    # NaN at bar 0 where prev_c is NaN (matches Pine's math.max with na = na)
    tr = np.where(np.isnan(prev_c), np.nan, np.maximum(tr1, np.maximum(tr2, tr3)))

    atr      = _pine_rma(tr,      dilen)
    plus_rma = _pine_rma(plus_dm, dilen)
    minus_rma= _pine_rma(minus_dm,dilen)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = 100.0 * plus_rma  / atr
        minus_di = 100.0 * minus_rma / atr

    # fixnan: forward-fill (Pine fixnan replaces na with last non-na value)
    def ffill(a):
        out = a.copy()
        for i in range(1, len(out)):
            if np.isnan(out[i]):
                out[i] = out[i-1]
        return out

    plus_di  = ffill(plus_di)
    minus_di = ffill(minus_di)

    dsum = plus_di + minus_di
    dx = np.where(dsum != 0, np.abs(plus_di - minus_di) / dsum * 100, 0.0)
    adx = _pine_rma(dx, adxlen)
    return pd.Series(adx, index=close.index)


def calc_adx_score(high, low, close, dilen=14, adxlen=14, smalen=20):
    """ADX score: 1 if ADX[1] > SMA(ADX[1], smalen), else 0."""
    shifted_h = high.shift(1)
    shifted_l = low.shift(1)
    shifted_c = close.shift(1)
    adx = calc_adx(shifted_h, shifted_l, shifted_c, dilen, adxlen)
    sma = adx.rolling(smalen).mean()
    score = (adx > sma).astype(int)
    return score


# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO SYSTEM FUNCTION  (compatible with PortfolioOptimiser)
# ─────────────────────────────────────────────────────────────────────

def gravity_arena_system(
    data_cache: dict,
    backtest_start:  str   = "2022-01-01",
    benchmark_name:  str   = "BTC",
    roc_length:      int   = 30,
    omega_length:    int   = 30,
    sortino_length:  int   = 30,
    zscore_length:   int   = 30,
    rsi_length:      int   = 30,
    rsi_threshold:   float = 50.0,
    beta_len:        int   = 40,
    alpha_len:       int   = 7,
    adx_smoothing:   int   = 14,
    di_length:       int   = 14,
    adx_sma_len:     int   = 20,
    roc_threshold:   float = 0.1,
    zscore_mean:     float = 1.5,
    top_n:           int   = 4,
) -> pd.Series:
    """
    Core Gravity Arena portfolio system.

    Parameters
    ----------
    data_cache      : dict {name: df} with at least 'close', 'high', 'low' columns.
                      Compatible with the data_cache produced by PortfolioOptimiser.
    backtest_start  : equity curve starts here (earlier data used as warmup).
    benchmark_name  : key in data_cache used as benchmark (e.g. 'BTC').
    top_n           : number of top-ranked assets to hold (equal weight).

    Returns
    -------
    pd.Series  — Top-N equity curve starting at 1.0.
    """
    available_assets = list(data_cache.keys())
    if benchmark_name not in data_cache:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in data_cache")

    # ── Build aligned price DataFrames ──────────────────────────────
    close_df = pd.DataFrame({n: data_cache[n]["close"] for n in available_assets})
    high_df  = pd.DataFrame({n: data_cache[n]["high"]  for n in available_assets})
    low_df   = pd.DataFrame({n: data_cache[n]["low"]   for n in available_assets})

    close_df = close_df.sort_index().ffill()
    high_df  = high_df.sort_index().ffill()
    low_df   = low_df.sort_index().ffill()

    common_idx = close_df.dropna(how="all").index
    close_df   = close_df.loc[common_idx]
    high_df    = high_df.loc[common_idx]
    low_df     = low_df.loc[common_idx]

    benchmark_close = close_df[benchmark_name]

    # ── Compute factors ─────────────────────────────────────────────
    roc_dict     = {}
    omega_dict   = {}
    sortino_dict = {}
    zscore_dict  = {}
    rsi_dict     = {}
    alpha_dict   = {}
    adx_dict     = {}

    for name in available_assets:
        c = close_df[name]
        h = high_df[name]
        l = low_df[name]
        roc_dict[name]     = calc_roc(c, roc_length)
        omega_dict[name]   = calc_omega_ratio(c, omega_length)
        sortino_dict[name] = calc_sortino_ratio(c, sortino_length)
        zscore_dict[name]  = calc_zscore(c, zscore_length)
        rsi_dict[name]     = calc_rsi(c, rsi_length)
        alpha_dict[name]   = calc_alpha(benchmark_close, c, beta_len, alpha_len)
        adx_dict[name]     = calc_adx_score(h, l, c, di_length, adx_smoothing, adx_sma_len)

    roc_df     = pd.DataFrame(roc_dict)
    omega_df   = pd.DataFrame(omega_dict)
    sortino_df = pd.DataFrame(sortino_dict)
    zscore_df  = pd.DataFrame(zscore_dict)
    rsi_df     = pd.DataFrame(rsi_dict)
    alpha_df   = pd.DataFrame(alpha_dict)
    adx_df     = pd.DataFrame(adx_dict)

    # ── Classifications ──────────────────────────────────────────────
    roc_class     = (roc_df > roc_threshold).astype(int)
    omega_median  = omega_df.median(axis=1)
    omega_class   = omega_df.ge(omega_median, axis=0).astype(int)
    sortino_med   = sortino_df.median(axis=1)
    sortino_class = sortino_df.ge(sortino_med, axis=0).astype(int)
    zscore_class  = (zscore_df > zscore_mean).astype(int)
    rsi_class     = (rsi_df > rsi_threshold).astype(int)
    alpha_class   = (alpha_df > 0).astype(int)
    adx_class     = adx_df.fillna(0).astype(int)

    total_scores = (roc_class + omega_class + sortino_class + zscore_class
                    + rsi_class + alpha_class + adx_class)

    eroc_df = close_df / close_df.shift(1) - 1

    # ── Bar-by-bar backtest ──────────────────────────────────────────
    backtest_ts = pd.Timestamp(backtest_start)
    dates = total_scores.index

    equity      = 1.0
    equity_vals = []
    equity_idx  = []
    prev_selected: list = []

    for date in dates:
        scores_today = total_scores.loc[date]
        alpha_today  = alpha_df.loc[date]
        candidates   = scores_today[scores_today == 7].index.tolist()
        candidates   = [n for n in candidates if not pd.isna(alpha_today.get(n, np.nan))]
        candidates.sort(key=lambda n: alpha_today[n], reverse=True)
        current_selected = candidates[:top_n]

        if date < backtest_ts:
            if current_selected:
                prev_selected = current_selected
            continue

        num_current = len(current_selected)
        if num_current > 0 and prev_selected:
            num_portfolio = min(top_n, num_current)
            weight        = 1.0 / num_portfolio
            portfolio_ret = 0.0
            for rank in range(num_portfolio):
                if rank < len(prev_selected):
                    name = prev_selected[rank]
                    ret  = eroc_df.loc[date, name] if name in eroc_df.columns else 0.0
                    if pd.isna(ret):
                        ret = 0.0
                    portfolio_ret += weight * ret
            equity *= (1 + portfolio_ret)

        equity_vals.append(equity)
        equity_idx.append(date)

        if current_selected:
            prev_selected = current_selected

    return pd.Series(equity_vals, index=equity_idx)


# ─────────────────────────────────────────────────────────────────────
# MAIN BACKTEST  (standalone, loads data itself)
# ─────────────────────────────────────────────────────────────────────
def run_backtest(bt=None, show_results=True, show_plot=True):
    if bt is None:
        bt = Backtester()

    if show_results:
        print("=" * 70)
        print("  GRAVITY ARENA  —  Python Backtest")
        print("=" * 70)
        print()

    # ── Load data ONCE ──────────────────────────────────────────────
    print("Loading data...")
    all_data = {}
    for name, symbol in ASSETS.items():
        if symbol is None:
            print(f"  Skipping {name} (no symbol mapped)")
            continue
        print(f"  Fetching {name} ({symbol})...")
        df = bt.get_data(symbol, TIMEFRAME, start_date, end_date)
        if df is not None and not df.empty and len(df) > 60:
            all_data[name] = df
        else:
            print(f"  -> Insufficient data for {name}")

    available_assets = [name for name in ASSET_NAMES if name in all_data]
    print(f"\nLoaded {len(available_assets)}/{len(ASSET_NAMES)} assets")

    if BENCHMARK_NAME not in all_data:
        print("ERROR: Benchmark (BTC) data not available. Aborting.")
        return

    # ── Run portfolio system ─────────────────────────────────────────
    eq_top4 = gravity_arena_system(
        all_data,
        backtest_start = BACKTEST_START,
        benchmark_name = BENCHMARK_NAME,
        roc_length     = ROC_LENGTH,
        omega_length   = OMEGA_LENGTH,
        sortino_length = SORTINO_LENGTH,
        zscore_length  = ZSCORE_LENGTH,
        rsi_length     = RSI_LENGTH,
        rsi_threshold  = RSI_THRESHOLD,
        beta_len       = BETA_LEN,
        alpha_len      = ALPHA_LEN,
        adx_smoothing  = ADX_SMOOTHING,
        di_length      = DI_LENGTH,
        adx_sma_len    = ADX_SMA_LEN,
        roc_threshold  = ROC_THRESHOLD,
        zscore_mean    = ZSCORE_MEAN,
        top_n          = 4,
    )
    eq_top1 = gravity_arena_system(
        all_data,
        backtest_start = BACKTEST_START,
        benchmark_name = BENCHMARK_NAME,
        roc_length     = ROC_LENGTH,
        omega_length   = OMEGA_LENGTH,
        sortino_length = SORTINO_LENGTH,
        zscore_length  = ZSCORE_LENGTH,
        rsi_length     = RSI_LENGTH,
        rsi_threshold  = RSI_THRESHOLD,
        beta_len       = BETA_LEN,
        alpha_len      = ALPHA_LEN,
        adx_smoothing  = ADX_SMOOTHING,
        di_length      = DI_LENGTH,
        adx_sma_len    = ADX_SMA_LEN,
        roc_threshold  = ROC_THRESHOLD,
        zscore_mean    = ZSCORE_MEAN,
        top_n          = 1,
    )

    eq_df = pd.DataFrame({"Top 1": eq_top1.values, "Top 4": eq_top4.values},
                         index=eq_top4.index)

    # Rebuild structures needed for display (factor tables, score table)
    close_df = pd.DataFrame({n: all_data[n]["close"] for n in available_assets})
    high_df  = pd.DataFrame({n: all_data[n]["high"]  for n in available_assets})
    low_df   = pd.DataFrame({n: all_data[n]["low"]   for n in available_assets})
    close_df = close_df.sort_index().ffill()
    high_df  = high_df.sort_index().ffill()
    low_df   = low_df.sort_index().ffill()
    common_idx = close_df.dropna(how="all").index
    close_df   = close_df.loc[common_idx]
    high_df    = high_df.loc[common_idx]
    low_df     = low_df.loc[common_idx]
    benchmark_close = close_df[BENCHMARK_NAME]

    roc_dict = {n: calc_roc(close_df[n], ROC_LENGTH) for n in available_assets}
    omega_dict = {n: calc_omega_ratio(close_df[n], OMEGA_LENGTH) for n in available_assets}
    sortino_dict = {n: calc_sortino_ratio(close_df[n], SORTINO_LENGTH) for n in available_assets}
    zscore_dict = {n: calc_zscore(close_df[n], ZSCORE_LENGTH) for n in available_assets}
    rsi_dict = {n: calc_rsi(close_df[n], RSI_LENGTH) for n in available_assets}
    alpha_dict = {n: calc_alpha(benchmark_close, close_df[n], BETA_LEN, ALPHA_LEN) for n in available_assets}
    adx_dict = {n: calc_adx_score(high_df[n], low_df[n], close_df[n], DI_LENGTH, ADX_SMOOTHING, ADX_SMA_LEN) for n in available_assets}

    roc_df     = pd.DataFrame(roc_dict)
    omega_df   = pd.DataFrame(omega_dict)
    sortino_df = pd.DataFrame(sortino_dict)
    zscore_df  = pd.DataFrame(zscore_dict)
    rsi_df     = pd.DataFrame(rsi_dict)
    alpha_df   = pd.DataFrame(alpha_dict)
    adx_df     = pd.DataFrame(adx_dict)

    roc_class     = (roc_df > ROC_THRESHOLD).astype(int)
    omega_median  = omega_df.median(axis=1)
    omega_class   = omega_df.ge(omega_median, axis=0).astype(int)
    sortino_med   = sortino_df.median(axis=1)
    sortino_class = sortino_df.ge(sortino_med, axis=0).astype(int)
    zscore_class  = (zscore_df > ZSCORE_MEAN).astype(int)
    rsi_class     = (rsi_df > RSI_THRESHOLD).astype(int)
    alpha_class   = (alpha_df > 0).astype(int)
    adx_class     = adx_df.fillna(0).astype(int)
    total_scores  = (roc_class + omega_class + sortino_class + zscore_class
                     + rsi_class + alpha_class + adx_class)

    # Reconstruct prev_selected (last qualifying bar's selection) for display
    prev_selected = []
    for date in total_scores.index:
        scores_today = total_scores.loc[date]
        alpha_today  = alpha_df.loc[date]
        candidates   = scores_today[scores_today == 7].index.tolist()
        candidates   = [n for n in candidates if not pd.isna(alpha_today.get(n, np.nan))]
        candidates.sort(key=lambda n: alpha_today[n], reverse=True)
        if candidates:
            prev_selected = candidates[:4]

    # ── BTC buy & hold ──
    backtest_ts = pd.Timestamp(BACKTEST_START)
    btc_start_price = close_df.loc[close_df.index >= backtest_ts, BENCHMARK_NAME].iloc[0]
    btc_bnh = close_df.loc[close_df.index >= backtest_ts, BENCHMARK_NAME] / btc_start_price
    btc_bnh = btc_bnh.reindex(eq_df.index, method="ffill")

    # ─────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────
    def compute_metrics(equity_series, name):
        eq = equity_series.values
        returns = np.diff(eq) / eq[:-1]
        returns = returns[~np.isnan(returns)]

        total_return = (eq[-1] / eq[0] - 1) * 100

        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1
        max_dd = np.min(dd) * 100

        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-10
        sharpe = mean_r / std_r * np.sqrt(365) if std_r > 0 else 0

        neg_returns = returns[returns < 0]
        neg_std = np.std(neg_returns, ddof=1) if len(neg_returns) > 1 else 1e-10
        sortino = mean_r / neg_std * np.sqrt(365) if neg_std > 0 else 0

        pos_sum = np.sum(returns[returns > 0])
        neg_sum = np.sum(np.abs(returns[returns < 0]))
        omega = pos_sum / neg_sum if neg_sum > 0 else np.inf

        return {
            "Strategy": name,
            "Total Return (%)": round(total_return, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Omega Ratio": round(omega, 2),
        }

    metrics_top1 = compute_metrics(eq_df["Top 1"], "Top 1")
    metrics_top4 = compute_metrics(eq_df["Top 4"], "Top 4")
    metrics_btc = compute_metrics(btc_bnh.dropna(), "BTC Buy & Hold")

    if show_results:
        # ── Print Results ──
        print()
        print("=" * 70)
        print("  GRAVITY ARENA METRICS")
        print("=" * 70)
        print(f"  Backtest Period : {BACKTEST_START} -> {eq_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Fee             : {FEE_PER_TRADE:.4%}/side ({2*FEE_PER_TRADE:.4%} round trip)")
        print(f"  Assets Tracked  : {len(available_assets)}")
        print(f"  Bars            : {len(eq_df)}")
        print()

        header = f"{'Strategy':<18} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Sortino':>9} {'Omega':>8}"
        print(header)
        print("-" * len(header))

        for m in [metrics_top1, metrics_top4, metrics_btc]:
            line = f"{m['Strategy']:<18} {m['Total Return (%)']:>9.2f}% {m['Max Drawdown (%)']:>9.2f}% {m['Sharpe Ratio']:>8.2f} {m['Sortino Ratio']:>9.2f} {m['Omega Ratio']:>8.2f}"
            print(line)

        print()
        print("-- Current Allocations --")
        print(f"  TOP PICKS : {', '.join(prev_selected) if prev_selected else 'None'}")
        print()

        # ── Print last day's score table ──
        last_date = total_scores.index[-1]
        if last_date >= backtest_ts:
            print("-- Final Bar Score Table --")
            scores_final = total_scores.loc[last_date]
            alpha_final = alpha_df.loc[last_date]
            roc_c = roc_class.loc[last_date]
            omega_c = omega_class.loc[last_date]
            sortino_c = sortino_class.loc[last_date]
            zscore_c = zscore_class.loc[last_date]
            rsi_c = rsi_class.loc[last_date]
            alpha_c = alpha_class.loc[last_date]
            adx_c = adx_class.loc[last_date]

            print(f"{'Asset':<8} {'ROC':>4} {'Omg':>4} {'Sort':>5} {'Z':>4} {'RSI':>4} {'Alp':>4} {'ADX':>4} {'Total':>6} {'Alpha':>10}")
            print("-" * 60)
            for name in available_assets:
                s = int(scores_final.get(name, 0))
                a = alpha_final.get(name, 0)
                marker = " << LONG" if name in prev_selected else ""
                print(f"{name:<8} {int(roc_c.get(name,0)):>4} {int(omega_c.get(name,0)):>4} {int(sortino_c.get(name,0)):>5} {int(zscore_c.get(name,0)):>4} {int(rsi_c.get(name,0)):>4} {int(alpha_c.get(name,0)):>4} {int(adx_c.get(name,0)):>4} {s:>6} {a:>10.5f}{marker}")
            print()

    if show_plot:
        # ─────────────────────────────────────────────────────────────────
        # PLOT
        # ─────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0a0a0a")

        # ── Equity curves ──
        ax1 = axes[0]
        ax1.set_facecolor("#0a0a0a")
        ax1.plot(eq_df.index, eq_df["Top 1"], color="#FF006E", linewidth=2, label=f"Top 1  {metrics_top1['Total Return (%)']:+.1f}%")
        ax1.plot(eq_df.index, eq_df["Top 4"], color="#3A86FF", linewidth=2, label=f"Top 4  {metrics_top4['Total Return (%)']:+.1f}%")
        ax1.plot(btc_bnh.index, btc_bnh.values, color="#FFD700", linewidth=1, alpha=0.5, linestyle="--", label=f"BTC B&H  {metrics_btc['Total Return (%)']:+.1f}%")
        ax1.set_yscale("log")
        ax1.axhline(1.0, color="gray", linewidth=0.5, alpha=0.3)
        ax1.set_title("GRAVITY ARENA  —  Equity Curves", color="white", fontsize=14, pad=12)
        ax1.set_ylabel("Equity (log scale)", color="white")
        ax1.legend(loc="upper left", fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.15)
        for spine in ax1.spines.values():
            spine.set_color("#333")

        # ── Drawdown ──
        ax2 = axes[1]
        ax2.set_facecolor("#0a0a0a")

        for col, color, lw in [("Top 1", "#FF006E", 1.5), ("Top 4", "#3A86FF", 1.5)]:
            eq = eq_df[col].values
            peak = np.maximum.accumulate(eq)
            dd = (eq / peak - 1) * 100
            ax2.fill_between(eq_df.index, dd, 0, alpha=0.15, color=color)
            ax2.plot(eq_df.index, dd, color=color, linewidth=lw, alpha=0.8)

        ax2.set_title("Drawdown (%)", color="white", fontsize=11, pad=8)
        ax2.set_ylabel("DD %", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, alpha=0.15)
        for spine in ax2.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
        plt.show()

    return eq_df


if __name__ == "__main__":
    bt = Backtester()
    run_backtest(bt)
