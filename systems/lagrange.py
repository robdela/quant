#!/usr/bin/env python3
"""
𝓖𝓡𝓐𝓥𝓘𝓣𝓨 𝓐𝓡𝓔𝓝𝓐 HEDGE
Ranks crypto assets on 3 factors (Sharpe, Omega, Alpha),
longs the strongest 3 and shorts the weakest 3.
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
BACKTEST_START = "2022-12-01"
TIMEFRAME = "1d"
LONG_SHORT_RATIO = "50/50"
FEE_PER_TRADE = 0.0005

start_date = "2023-01-01"
end_date = str(datetime.now().date())

SHARPE_LENGTH = 40
OMEGA_LENGTH = 40
BETA_LEN = 40
ALPHA_LEN = 10

ASSETS = {
    "ETH":   "ETH/USDT",
    "BNB":   "BNB/USDT",
    "SOL":   "SOL/USDT",
    "ADA":   "ADA/USDT",
    "XRP":   "XRP/USDT",
    "LUNA":  "LUNA/USDT",
    "DOT":   "DOT/USDT",
    "AVAX":  "AVAX/USDT",
    "DOGE":  "DOGE/USDT",
    "SHIB":  "SHIB/USDT",
    "CRO":   "CRO/USDT",
    "ALGO":  "ALGO/USDT",
    "UNI":   "UNI/USDT",
    "LTC":   "LTC/USDT",
    "LINK":  "LINK/USDT",
    "NEAR":  "NEAR/USDT",
    "BCH":   "BCH/USDT",
    "ATOM":  "ATOM/USDT",
    "TRX":   "TRX/USDT",
    "XLM":   "XLM/USDT",
    "FTM":   "FTM/USDT",
    "MANA":  "MANA/USDT",
    "AXS":   "AXS/USDT",
    "VET":   "VET/USDT",
    "FTT":   "FTT/USDT",
    "SAND":  "SAND/USDT",
    "HBAR":  "HBAR/USDT",
    "FIL":   "FIL/USDT",
    "THETA": "THETA/USDT",
    "ICP":   "ICP/USDT",
    "EGLD":  "EGLD/USDT",
    "ETC":   "ETC/USDT",
    "XMR":   "XMR/USDT",
    "HNT":   "HNT/USDT",
    "ONDO":  "ONDO/USDT",
    "XTZ":   "XTZ/USDT",
    "PAXG":  "PAXG/USDT",
    "AAVE":  "AAVE/USDT",
    "BTC":   "BTC/USDT",
}

ASSET_NAMES = list(ASSETS.keys())
BENCHMARK_NAME = "BTC"


# ─────────────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def calc_sharpe(close, length):
    """Rolling Sharpe ratio on close[1]/close[2]-1 returns."""
    returns = close.shift(1) / close.shift(2) - 1
    sharpe = pd.Series(np.nan, index=close.index)
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
                sharpe.iloc[i] = (mean_r / std_r) * np.sqrt(252)

    return sharpe


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


def calc_beta(benchmark_close, asset_close, length):
    """Log-return beta of asset vs benchmark."""
    asset_ret = np.log(asset_close.shift(1) / asset_close.shift(2))
    bench_ret = np.log(benchmark_close.shift(1) / benchmark_close.shift(2))
    avg_a = asset_ret.rolling(length).mean()
    avg_b = bench_ret.rolling(length).mean()
    cov = ((asset_ret - avg_a) * (bench_ret - avg_b)).rolling(length).mean()
    var_b = bench_ret.rolling(length).var()
    return cov / var_b


def calc_alpha(benchmark_close, asset_close, beta_len, alpha_len):
    """Alpha = avg_asset_return - beta * avg_benchmark_return (averaged across 3 windows)."""
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

    return avg_a - beta_avg * avg_b


# ─────────────────────────────────────────────────────────────────────
# MAIN BACKTEST
# ─────────────────────────────────────────────────────────────────────
def run_backtest(bt=None, show_results=True, show_plot=True,
                 sharpe_length=None, omega_length=None,
                 beta_len=None, alpha_len=None,
                 long_short_ratio=None, fee_per_trade=None,
                 n_longs=None, n_shorts=None):
    _sharpe_length    = sharpe_length    if sharpe_length    is not None else SHARPE_LENGTH
    _omega_length     = omega_length     if omega_length     is not None else OMEGA_LENGTH
    _beta_len         = beta_len         if beta_len         is not None else BETA_LEN
    _alpha_len        = alpha_len        if alpha_len        is not None else ALPHA_LEN
    _long_short_ratio = long_short_ratio if long_short_ratio is not None else LONG_SHORT_RATIO
    _fee_per_trade    = fee_per_trade    if fee_per_trade    is not None else FEE_PER_TRADE
    _n_longs          = n_longs          if n_longs          is not None else 3
    _n_shorts         = n_shorts         if n_shorts         is not None else 3

    if bt is None:
        bt = Backtester()

    if show_results:
        print("=" * 70)
        print("  𝓖𝓡𝓐𝓥𝓘𝓣𝓨 𝓐𝓡𝓔𝓝𝓐 HEDGE  —  Python Backtest")
        print("=" * 70)
        print()

    # ── Load data ──
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

    # ── Build aligned price DataFrame ──
    close_df = pd.DataFrame({name: all_data[name]["close"] for name in available_assets})
    close_df = close_df.sort_index().ffill()
    common_idx = close_df.dropna(how="all").index
    close_df = close_df.loc[common_idx]

    benchmark_close = close_df[BENCHMARK_NAME]

    # ── Compute factors ──
    print("Computing factors...")

    sharpe_dict = {}
    omega_dict = {}
    alpha_dict = {}

    for name in available_assets:
        c = close_df[name]
        sharpe_dict[name] = calc_sharpe(c, _sharpe_length)
        omega_dict[name] = calc_omega_ratio(c, _omega_length)
        alpha_dict[name] = calc_alpha(benchmark_close, c, _beta_len, _alpha_len)

    sharpe_df = pd.DataFrame(sharpe_dict)
    omega_df = pd.DataFrame(omega_dict)
    alpha_df = pd.DataFrame(alpha_dict)

    # ── Cross-sectional ranking ──
    # Rank each factor across assets (higher value = higher rank)
    print("Ranking...")
    sharpe_rank = sharpe_df.rank(axis=1, ascending=True, na_option="bottom")
    omega_rank = omega_df.rank(axis=1, ascending=True, na_option="bottom")
    alpha_rank = alpha_df.rank(axis=1, ascending=True, na_option="bottom")

    # Composite score = sum of ranks (higher = stronger)
    composite_score = sharpe_rank + omega_rank + alpha_rank

    # ── Parse long/short ratio ──
    ratio_map = {
        "40/60": 0.40, "45/55": 0.45, "50/50": 0.50,
        "55/45": 0.55, "60/40": 0.60, "65/35": 0.65, "70/30": 0.70
    }
    if isinstance(_long_short_ratio, (int, float)):
        long_pct = float(_long_short_ratio)
    else:
        long_pct = ratio_map.get(_long_short_ratio, 0.50)
    short_pct = 1.0 - long_pct

    # ── Backtest loop ──
    print("Running backtest...")

    backtest_start = pd.Timestamp(BACKTEST_START)
    dates = composite_score.index

    equity_hedge = 1.0
    equity_long_only = 1.0
    equity_short_only = 1.0

    long_positions = []
    short_positions = []

    equity_hedge_series = []
    equity_long_series = []
    equity_short_series = []
    equity_dates = []

    last_longs = []
    last_shorts = []

    for i, date in enumerate(dates):
        if date < backtest_start:
            continue

        scores_today = composite_score.loc[date].dropna()
        if scores_today.empty:
            continue

        # Top N = longs, bottom N = shorts
        sorted_assets = scores_today.sort_values(ascending=False)
        selected_longs = sorted_assets.index[:_n_longs].tolist()
        selected_shorts = sorted_assets.index[-_n_shorts:].tolist()

        num_longs = len(selected_longs)
        num_shorts = len(selected_shorts)

        prev_long_names = {name for (name, _, _) in long_positions}
        prev_short_names = {name for (name, _, _) in short_positions}
        new_long_names = set(selected_longs)
        new_short_names = set(selected_shorts)

        longs_to_close = prev_long_names - new_long_names
        longs_to_open = new_long_names - prev_long_names
        shorts_to_close = prev_short_names - new_short_names
        shorts_to_open = new_short_names - prev_short_names

        # ── Mark-to-market ──
        long_pnl = 0.0
        long_only_pnl = 0.0
        num_active_longs = len(long_positions)
        long_only_w = 1.0 / num_active_longs if num_active_longs > 0 else 0.0

        for (name, entry_price, weight) in long_positions:
            cur_price = close_df.loc[date, name] if name in close_df.columns else np.nan
            if not pd.isna(cur_price) and entry_price > 0:
                ret = cur_price / entry_price - 1
                fees = 2 * _fee_per_trade if name in longs_to_close else 0.0
                long_pnl += (ret - fees) * weight
                long_only_pnl += (ret - fees) * long_only_w

        short_pnl = 0.0
        short_only_pnl = 0.0
        num_active_shorts = len(short_positions)
        short_only_w = 1.0 / num_active_shorts if num_active_shorts > 0 else 0.0

        for (name, entry_price, weight) in short_positions:
            cur_price = close_df.loc[date, name] if name in close_df.columns else np.nan
            if not pd.isna(cur_price) and entry_price > 0:
                ret = entry_price / cur_price - 1
                fees = 2 * _fee_per_trade if name in shorts_to_close else 0.0
                short_pnl += (ret - fees) * weight
                short_only_pnl += (ret - fees) * short_only_w

        for name in longs_to_open:
            w = long_pct / num_longs if num_longs > 0 else 0.0
            long_pnl -= _fee_per_trade * w
            long_only_pnl -= _fee_per_trade * (1.0 / num_longs if num_longs > 0 else 0.0)
        for name in shorts_to_open:
            w = short_pct / num_shorts if num_shorts > 0 else 0.0
            short_pnl -= _fee_per_trade * w
            short_only_pnl -= _fee_per_trade * (1.0 / num_shorts if num_shorts > 0 else 0.0)

        equity_hedge *= (1 + long_pnl + short_pnl)
        equity_long_only *= (1 + long_only_pnl)
        equity_short_only *= (1 + short_only_pnl)

        equity_hedge_series.append(equity_hedge)
        equity_long_series.append(equity_long_only)
        equity_short_series.append(equity_short_only)
        equity_dates.append(date)

        # ── Reset positions at current close ──
        long_positions = []
        for name in selected_longs:
            price = close_df.loc[date, name]
            w = long_pct / num_longs if num_longs > 0 else 0.0
            if not pd.isna(price):
                long_positions.append((name, price, w))

        short_positions = []
        for name in selected_shorts:
            price = close_df.loc[date, name]
            w = short_pct / num_shorts if num_shorts > 0 else 0.0
            if not pd.isna(price):
                short_positions.append((name, price, w))

        last_longs = selected_longs
        last_shorts = selected_shorts

    # ── Build equity DataFrames ──
    eq_df = pd.DataFrame({
        "Hedge (L+S)": equity_hedge_series,
        "Long Only": equity_long_series,
        "Short Only": equity_short_series,
    }, index=equity_dates)

    btc_start_price = close_df.loc[close_df.index >= backtest_start, BENCHMARK_NAME].iloc[0]
    btc_bnh = close_df.loc[close_df.index >= backtest_start, BENCHMARK_NAME] / btc_start_price
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

    metrics_hedge = compute_metrics(eq_df["Hedge (L+S)"], "Hedge (L+S)")
    metrics_long = compute_metrics(eq_df["Long Only"], "Long Only")
    metrics_short = compute_metrics(eq_df["Short Only"], "Short Only")
    metrics_btc = compute_metrics(btc_bnh.dropna(), "BTC Buy & Hold")

    if show_results:
        print()
        print("=" * 70)
        print("  𝓖𝓡𝓐𝓥𝓘𝓣𝓨 𝓐𝓡𝓔𝓝𝓐 𝓜𝓔𝓣𝓡𝓘𝓒𝓢")
        print("=" * 70)
        print(f"  Backtest Period : {BACKTEST_START} → {eq_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  L/S Ratio       : {_long_short_ratio}")
        print(f"  Fee             : {_fee_per_trade:.4%}/side ({2*_fee_per_trade:.4%} round trip)")
        print(f"  Assets Tracked  : {len(available_assets)}")
        print(f"  Bars            : {len(eq_df)}")
        print()

        header = f"{'Strategy':<18} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Sortino':>9} {'Omega':>8}"
        print(header)
        print("-" * len(header))

        for m in [metrics_hedge, metrics_long, metrics_short, metrics_btc]:
            line = f"{m['Strategy']:<18} {m['Total Return (%)']:>9.2f}% {m['Max Drawdown (%)']:>9.2f}% {m['Sharpe Ratio']:>8.2f} {m['Sortino Ratio']:>9.2f} {m['Omega Ratio']:>8.2f}"
            print(line)

        print()
        print("── Current Allocations ──")
        print(f"  LONG  : {', '.join(last_longs) if last_longs else 'None'}")
        print(f"  SHORT : {', '.join(last_shorts) if last_shorts else 'None'}")
        print()

        # ── Print last day's score table ──
        last_date = dates[-1]
        if last_date >= backtest_start:
            print("── Final Bar Rankings ──")
            sharpe_final = sharpe_df.loc[last_date]
            omega_final = omega_df.loc[last_date]
            alpha_final = alpha_df.loc[last_date]
            score_final = composite_score.loc[last_date]

            print(f"{'Asset':<8} {'Sharpe':>8} {'Omega':>8} {'Alpha':>10} {'Score':>8}")
            print("-" * 48)
            ranked = score_final.dropna().sort_values(ascending=False)
            for name in ranked.index:
                sh = sharpe_final.get(name, 0)
                om = omega_final.get(name, 0)
                al = alpha_final.get(name, 0)
                sc = score_final.get(name, 0)
                marker = " ◄ LONG" if name in last_longs else (" ◄ SHORT" if name in last_shorts else "")
                print(f"{name:<8} {sh:>8.2f} {om:>8.2f} {al:>10.5f} {sc:>8.1f}{marker}")
            print()

    if show_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#0a0a0a")

        ax1 = axes[0]
        ax1.set_facecolor("#0a0a0a")
        ax1.plot(eq_df.index, eq_df["Hedge (L+S)"], color="#FFD700", linewidth=2, label=f"Hedge (L+S)  {metrics_hedge['Total Return (%)']:+.1f}%")
        ax1.plot(eq_df.index, eq_df["Long Only"], color="#FF006E", linewidth=1.2, alpha=0.8, label=f"Long Only  {metrics_long['Total Return (%)']:+.1f}%")
        ax1.plot(eq_df.index, eq_df["Short Only"], color="#8338EC", linewidth=1.2, alpha=0.8, label=f"Short Only  {metrics_short['Total Return (%)']:+.1f}%")
        ax1.plot(btc_bnh.index, btc_bnh.values, color="#3A86FF", linewidth=1, alpha=0.5, linestyle="--", label=f"BTC B&H  {metrics_btc['Total Return (%)']:+.1f}%")
        ax1.axhline(1.0, color="gray", linewidth=0.5, alpha=0.3)
        ax1.set_title("GRAVITY ARENA HEDGE  —  Equity Curves", color="white", fontsize=14, pad=12)
        ax1.set_ylabel("Equity", color="white")
        ax1.legend(loc="upper left", fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.15)
        for spine in ax1.spines.values():
            spine.set_color("#333")

        ax2 = axes[1]
        ax2.set_facecolor("#0a0a0a")

        for col, color, lw in [("Hedge (L+S)", "#FFD700", 1.5), ("Long Only", "#FF006E", 1), ("Short Only", "#8338EC", 1)]:
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
