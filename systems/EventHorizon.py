"""
Event Horizon — Relative Strength Position Sizing System
Port of the TradingView Pine Script v6 indicator to Python3.

Uses genesis_v1 (TPI) and quark strategies from backtest.py.
"""

#    ╱|、
#  (˚ˎ 。7
#   |、˜〵
#   じしˍ,)ノ

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.backtest import Backtester
from strategies.genesis_v1 import genesis_v1
from strategies.quark import quark

# ─── Configuration ──────────────────────────────────────────────────────

TICKERS = {
    'BTC':  'BTC/USDT',
    'ETH':  'ETH/USDT',
    'SOL':  'SOL/USDT',
    'BNB':  'BNB/USDT',
    'DOGE': 'DOGE/USDT',
    'XRP':  'XRP/USDT',
    'ADA':  'ADA/USDT',
    'SUI':  'SUI/USDT',
    'LINK':  'LINK/USDT',
    'XLM': 'XLM/USDT',
    # 'UNI':  'UNI/USDT',
    # 'LTC':  'LTC/USDT',
    # 'AVAX': 'AVAX/USDT',
    # 'DOT': 'DOT/USDT',
    # 'ATOM': 'ATOM/USDT',
    # 'ETC': 'ETC/USDT',
    # 'XLM': 'XLM/USDT'



}

# Strategy used for each ticker's individual trend
STRATEGY_MAP = {
    'BTC': 'tpi', 'ETH': 'tpi', 'SOL': 'tpi', 'BNB': 'tpi',
    'DOGE': 'quark', 'XRP': 'quark', 'ADA': 'quark',
    'SUI': 'tpi', 'LINK': 'tpi', 'XLM': 'quark', 'LTC': 'quark', 'DOT': 'quark', 'SHIB': 'quark', 'UNI': 'quark',
    'TON': 'tpi', 'XMR': 'quark', 'AVAX': 'quark', 'ATOM': 'quark', 'ETC': 'quark'
}

# Only these tickers count for the "any_positive_trend" filter
MAJOR_TICKERS = ['BTC', 'ETH', 'BNB']


class EventHorizon:

    def __init__(self, timeframe='1d', start_date='2025-01-01', end_date=None,
                 exchange='binance', include_fees=False, fee_amount=0.01,
                 use_gold=False, use_leverage=False, lev_mult=1.2):
        self.bt = Backtester(exchange=exchange)
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date or str(datetime.now().date())
        self.include_fees = include_fees
        self.fee_amount = fee_amount
        self.use_gold = use_gold
        self.use_leverage = use_leverage
        self.lev_mult = lev_mult

        self.data = {}
        self.trends = {}
        self.rsps_scores = {}
        self.alphas = {}

        self.top_asset = None
        self.second_asset = None
        self.highest_alpha_asset = None
        self.raw_rsps_equity = None
        self.confidence = None
        self.final_equity = None
        self.rsps_running = None
        self.held_asset = None
        self.market_tpi = None

    # ─── Data ──────────────────────────────────────────────────

    def _fetch_data(self):
        warmup_start = str(
            (pd.Timestamp(self.start_date) - pd.Timedelta(days=150)).date()
        )
        print('\n  Loading data...')
        for name, symbol in TICKERS.items():
            print(f'  → {name} ({symbol})')
            self.data[name] = self.bt.get_data(
                symbol, self.timeframe,
                start_date=warmup_start, end_date=self.end_date,
            )

    def _align_data(self):
        # Find the latest start date across all tickers and warn about short ones
        starts = {name: df.index[0] for name, df in self.data.items() if len(df) > 0}
        latest_start = max(starts.values())
        for name, start in starts.items():
            if start > pd.Timestamp(self.start_date):
                print(f'  ⚠  {name} data starts {start.date()} — limits common window')

        common_idx = None
        for df in self.data.values():
            idx = df.index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        if len(common_idx) == 0:
            culprits = [n for n, s in starts.items() if s == latest_start]
            raise ValueError(
                f'No overlapping data after alignment. '
                f'Likely culprit(s): {culprits}. '
                f'Remove them from TICKERS or adjust start_date.'
            )

        for name in self.data:
            self.data[name] = self.data[name].loc[common_idx]

    # ─── Individual Trends ─────────────────────────────────────

    @staticmethod
    def _trend_signal(df, strategy):
        """Run a strategy and return 0/1 trend (Pine TPI/Quark wrapper)."""
        result = genesis_v1(df.copy()) if strategy == 'tpi' else quark(df.copy())
        return (result['signal'] == 1).astype(int)

    def _compute_all_trends(self):
        print('\n  Computing individual trends...')
        for name in TICKERS:
            print(f'  → {name}')
            self.trends[name] = self._trend_signal(
                self.data[name], STRATEGY_MAP[name]
            )

    # ─── Tournament (RSPS) ─────────────────────────────────────

    def _tpi_on_ratio(self, df_a, df_b):
        """Run genesis_v1 on the ratio of two assets → 0 or 1 per bar."""
        ratio_df = pd.DataFrame({
            'open':   df_a['open']   / df_b['open'],
            'high':   df_a['high']   / df_b['high'],
            'low':    df_a['low']    / df_b['low'],
            'close':  df_a['close']  / df_b['close'],
            'volume': df_a['volume'],
        }, index=df_a.index)
        ratio_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        ratio_df.ffill(inplace=True)
        ratio_df.bfill(inplace=True)

        result = genesis_v1(ratio_df)
        # TPI wrapper: signal 1 → 1, signal -1 → 0
        return (result['signal'] == 1).astype(int)

    def _run_tournament(self):
        print('\n  Running RSPS tournament...')
        names = list(TICKERS.keys())
        n = len(names)
        idx = self.data[names[0]].index
        scores = {name: pd.Series(0, index=idx, dtype=int) for name in names}

        total = n * n
        log_every = max(1, total // 10)
        done = 0
        for name_a in names:
            for name_b in names:
                done += 1
                if done % log_every == 0:
                    print(f'    matchup {done}/{total}')
                scores[name_a] = scores[name_a] + self._tpi_on_ratio(
                    self.data[name_a], self.data[name_b]
                )
        self.rsps_scores = scores
        print('  Tournament complete.')

    def _find_top_performers(self):
        score_df = pd.DataFrame(self.rsps_scores)

        def _top_two(row):
            s = row.sort_values(ascending=False)
            return s.index[0], s.index[1]

        tops = score_df.apply(_top_two, axis=1, result_type='expand')
        self.top_asset = tops.iloc[:, 0]
        self.second_asset = tops.iloc[:, 1]

    # ─── Alpha ─────────────────────────────────────────────────

    def _calc_benchmark(self):
        """Equal-weight normalised basket as TOTAL proxy."""
        names = list(TICKERS.keys())
        close_sum = high_sum = low_sum = None
        for name in names:
            base = self.data[name]['close'].iloc[0]
            c = self.data[name]['close'] / base
            h = self.data[name]['high']  / base
            l = self.data[name]['low']   / base
            if close_sum is None:
                close_sum, high_sum, low_sum = c, h, l
            else:
                close_sum += c; high_sum += h; low_sum += l
        k = len(names)
        return close_sum / k, high_sum / k, low_sum / k

    @staticmethod
    def _rolling_beta(bench_close, src_close, length):
        pr = np.log(src_close / src_close.shift(1))
        br = np.log(bench_close / bench_close.shift(1))
        avg_pr = pr.rolling(length).mean()
        avg_br = br.rolling(length).mean()
        cov = ((pr - avg_pr) * (br - avg_br)).rolling(length).mean()
        var = br.rolling(length).var()
        return cov / (var + 1e-10)

    def _alpha_series(self, bench_close, src_close,
                      beta_len=30, alpha_len=7):
        pr = np.log(src_close / src_close.shift(1))
        br = np.log(bench_close / bench_close.shift(1))

        beta = (self._rolling_beta(bench_close, src_close, beta_len) +
                self._rolling_beta(bench_close, src_close, beta_len + 9) +
                self._rolling_beta(bench_close, src_close, beta_len - 9)) / 3

        avg_pr = (pr.rolling(alpha_len).mean() +
                  pr.rolling(alpha_len + 3).mean() +
                  pr.rolling(alpha_len - 3).mean()) / 3
        avg_br = (br.rolling(alpha_len).mean() +
                  br.rolling(alpha_len + 3).mean() +
                  br.rolling(alpha_len - 3).mean()) / 3

        return avg_pr - beta * avg_br

    def _compute_all_alphas(self):
        print('\n  Computing alpha values...')
        bench_close, _, _ = self._calc_benchmark()
        for name in TICKERS:
            self.alphas[name] = self._alpha_series(
                bench_close.shift(1), self.data[name]['close'].shift(1)
            )

    def _find_highest_alpha(self):
        alpha_df = pd.DataFrame(self.alphas)
        # idxmax returns NaN for all-NaN rows; fill with first ticker as fallback
        self.highest_alpha_asset = alpha_df.idxmax(axis=1).fillna(list(TICKERS.keys())[0])

    # ─── Raw Equity & Confidence ───────────────────────────────

    def _calc_raw_rsps_equity(self):
        """Unfiltered equity: always follow the RSPS top asset."""
        idx = self.data['BTC'].index
        mask = np.array(idx >= pd.Timestamp(self.start_date))
        rocs = {n: self.data[n]['close'].pct_change() for n in TICKERS}

        eq = np.ones(len(idx))
        for i in range(1, len(idx)):
            if not mask[i]:
                eq[i] = eq[i - 1]
                continue
            top = self.top_asset.iloc[i - 1]
            r = rocs[top].iloc[i] if top in rocs else 0.0
            eq[i] = eq[i - 1] * (1 + (r if np.isfinite(r) else 0.0))
        self.raw_rsps_equity = pd.Series(eq, index=idx)

    def _calc_confidence_score(self):
        """
        Approximate don.ivalpiniEQ / don.COMSUBINeq on the equity curve.
        ivAlpini proxy  → EMA(10) vs EMA(30) crossover
        COMSUBIN proxy  → ROC(14) vs its MA(14)
        """
        eq = self.raw_rsps_equity

        # Proxy 1: EMA crossover
        fast = eq.ewm(span=10, adjust=False).mean()
        slow = eq.ewm(span=30, adjust=False).mean()
        iv = pd.Series(0, index=eq.index, dtype=int)
        iv[fast > slow] = 1
        iv[fast < slow] = -1

        # Proxy 2: momentum
        roc = eq.pct_change(14)
        roc_ma = roc.rolling(14).mean()
        com = pd.Series(0, index=eq.index, dtype=int)
        com[roc > roc_ma] = 1
        com[roc < roc_ma] = -1

        combined = iv + com
        conf = np.zeros(len(eq), dtype=int)
        for i in range(1, len(conf)):
            if combined.iloc[i] > 0:
                conf[i] = 1
            elif combined.iloc[i] < 0:
                conf[i] = -1
            else:
                conf[i] = conf[i - 1]
        self.confidence = pd.Series(conf, index=eq.index)

    # ─── Market TPI ────────────────────────────────────────────

    def _calc_market_tpi(self):
        """Average of all individual trends (shifted 1 bar)."""
        bench_close, bench_high, bench_low = self._calc_benchmark()
        bench_df = pd.DataFrame({
            'open': bench_close,
            'high': bench_high,
            'low': bench_low,
            'close': bench_close,
            'volume': self.data['BTC']['volume'],
        }, index=bench_close.index)
        total_trend = self._trend_signal(bench_df, 'tpi')

        all_trends = pd.DataFrame(self.trends)
        all_trends['TOTAL'] = total_trend
        self.market_tpi = all_trends.shift(1).mean(axis=1)

    # ─── Final Equity ──────────────────────────────────────────

    def _calc_final_equity(self):
        idx = self.data['BTC'].index
        mask = np.array(idx >= pd.Timestamp(self.start_date))
        rocs = {n: self.data[n]['close'].pct_change() for n in TICKERS}

        n = len(idx)
        eq = np.ones(n)
        running = np.ones(n, dtype=bool)
        held = np.empty(n, dtype=object)
        held[:] = 'CASH'

        for i in range(1, n):
            if not mask[i]:
                eq[i] = eq[i - 1]
                continue

            # Previous bar values
            any_pos = any(
                self.trends[m].iloc[i - 1] == 1 for m in MAJOR_TICKERS
            )
            top = self.top_asset.iloc[i - 1]
            second = self.second_asset.iloc[i - 1]
            conf = self.confidence.iloc[i - 1] == 1
            alpha_top = self.highest_alpha_asset.iloc[i - 1]

            # Guard against NaN / missing ticker names
            if not isinstance(top, str) or top not in self.trends:
                eq[i] = eq[i - 1]
                running[i] = False
                continue

            # Allocation: 100/0 if top is BTC/ETH/SOL, else 80/20
            if top in ('BTC', 'ETH', 'SOL'):
                top_alloc, sec_alloc = 1.0, 0.0
            else:
                top_alloc, sec_alloc = 0.8, 0.2

            if any_pos and conf:
                # ── RSPS mode ──
                top_in_trend = self.trends[top].iloc[i - 1] == 1
                if not top_in_trend:
                    running[i] = False
                    eq[i] = eq[i - 1]
                    continue

                top_ret = rocs[top].iloc[i]
                top_ret = top_ret if np.isfinite(top_ret) else 0.0
                held[i] = top

                sec_ret = 0.0
                if sec_alloc > 0:
                    sec_in_trend = self.trends[second].iloc[i - 1] == 1
                    if sec_in_trend:
                        r = rocs[second].iloc[i]
                        sec_ret = r if np.isfinite(r) else 0.0

                port_ret = top_alloc * top_ret + sec_alloc * sec_ret

                fees = 0.0
                if self.include_fees and i >= 2:
                    if self.top_asset.iloc[i - 1] != self.top_asset.iloc[i - 2]:
                        fees = self.fee_amount

                eq[i] = eq[i - 1] * (1 + port_ret) * (1 - fees)
                running[i] = True

            elif any_pos and not conf:
                # ── Alpha mode ──
                if not isinstance(alpha_top, str) or alpha_top not in self.trends:
                    running[i] = False
                    eq[i] = eq[i - 1]
                    continue
                alpha_in_trend = self.trends[alpha_top].iloc[i - 1] == 1
                if not alpha_in_trend:
                    running[i] = False
                    eq[i] = eq[i - 1]
                    continue

                top_ret = rocs[alpha_top].iloc[i]
                top_ret = top_ret if np.isfinite(top_ret) else 0.0
                held[i] = alpha_top

                fees = 0.0
                if self.include_fees and i >= 2:
                    prev_alpha = self.highest_alpha_asset.iloc[i - 2]
                    if alpha_top != prev_alpha:
                        fees = self.fee_amount

                eq[i] = eq[i - 1] * (1 + top_ret) * (1 - fees)
                running[i] = True

            else:
                # ── Cash ──
                running[i] = False
                eq[i] = eq[i - 1]

        self.final_equity = pd.Series(eq, index=idx)
        self.rsps_running = pd.Series(running, index=idx)
        self.held_asset = pd.Series(held, index=idx)

    # ─── Metrics ───────────────────────────────────────────────

    def _compute_metrics(self, equity):
        eq = equity[equity.index >= pd.Timestamp(self.start_date)]
        if len(eq) < 2:
            return {}
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        daily_ret = eq.pct_change().dropna()
        days = (eq.index[-1] - eq.index[0]).days
        cagr = (1 + total_ret) ** (365 / max(days, 1)) - 1 if days > 0 else 0

        dd = eq / eq.cummax() - 1
        max_dd = dd.min()

        std = daily_ret.std()
        sharpe = np.sqrt(365) * daily_ret.mean() / std if std > 0 else 0.0

        neg_std = daily_ret[daily_ret < 0].std()
        sortino = np.sqrt(365) * daily_ret.mean() / neg_std if neg_std > 0 else 0.0

        pos = daily_ret[daily_ret > 0].sum()
        neg = daily_ret[daily_ret < 0].sum()
        omega = pos / abs(neg) if neg != 0 else float('inf')

        return {
            'total_return': total_ret, 'cagr': cagr,
            'max_drawdown': max_dd,    'sharpe': sharpe,
            'sortino': sortino,        'omega': omega,
        }

    # ─── Display ───────────────────────────────────────────────

    def _print_results(self):
        W = 70
        print(f'\n┌{"─" * (W - 2)}┐')
        print(f'│  {"𝓔𝓥𝓔𝓝𝓣 𝓗𝓞𝓡𝓘𝓩𝓞𝓝":<{W - 4}}│')
        print(f'│  {f"{self.start_date}  →  {self.end_date}":<{W - 4}}│')
        print(f'└{"─" * (W - 2)}┘')

        last = -1
        top = self.top_asset.iloc[last]
        second = self.second_asset.iloc[last]
        conf = 'High' if self.confidence.iloc[last] == 1 else 'Low'
        held = self.held_asset.iloc[last]
        running = self.rsps_running.iloc[last]
        mode = ('RSPS' if running and conf == 'High'
                else 'ALPHA' if running else 'CASH')

        print(f'\n  Current State:')
        print(f'  {"System:":<20} {mode}')
        print(f'  {"Confidence:":<20} {conf}')
        print(f'  {"Holding:":<20} {held}')
        print(f'  {"RSPS Top:":<20} {top}')
        print(f'  {"RSPS Second:":<20} {second}')
        print(f'  {"Alpha Top:":<20} {self.highest_alpha_asset.iloc[last]}')

        print(f'\n  Trends:')
        for name in TICKERS:
            val = self.trends[name].iloc[last]
            print(f'    {name + ":":<12} {"LONG" if val == 1 else "SHORT"}')

        if self.market_tpi is not None:
            print(f'\n  Market TPI:      {self.market_tpi.iloc[last]:.2f}')

        for label, eq in [('RSPS Equity (filtered)', self.final_equity),
                          ('Raw RSPS (unfiltered)',   self.raw_rsps_equity)]:
            m = self._compute_metrics(eq)
            if not m:
                continue
            print(f'\n  {label}:')
            print(f'    {"Return:":<16} {m["total_return"]:+.2%}')
            print(f'    {"CAGR:":<16} {m["cagr"]:+.2%}')
            print(f'    {"Max DD:":<16} {m["max_drawdown"]:.2%}')
            print(f'    {"Sharpe:":<16} {m["sharpe"]:.2f}')
            print(f'    {"Sortino:":<16} {m["sortino"]:.2f}')
            print(f'    {"Omega:":<16} {m["omega"]:.2f}')

    def _plot(self):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                 height_ratios=[3, 1, 1],
                                 gridspec_kw={'hspace': 0.35})
        mask = self.final_equity.index >= pd.Timestamp(self.start_date)
        eq = self.final_equity[mask]
        raw = self.raw_rsps_equity[mask]

        # ── Equity ──
        ax = axes[0]
        ax.plot(eq.index, eq.values, lw=2, color='#FF006E', label='RSPS Equity')
        ax.plot(raw.index, raw.values, lw=1, color='gray', alpha=.5,
                label='Raw RSPS (unfiltered)')
        btc = self.data['BTC']['close'][mask]
        ax.plot(btc.index, (btc / btc.iloc[0]).values, lw=1, color='orange',
                alpha=.5, label='BTC Buy & Hold')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, alpha=.2, ls='--')
        ax.set_title('Event Horizon · Equity', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)

        # ── Confidence ──
        ax2 = axes[1]
        conf = self.confidence[mask]
        ax2.fill_between(conf.index, conf.values, 0, alpha=.3,
                         where=conf > 0, color='green', label='High')
        ax2.fill_between(conf.index, conf.values, 0, alpha=.3,
                         where=conf < 0, color='red', label='Low')
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_title('Confidence Score', fontsize=12)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=.2, ls='--')

        # ── Market TPI ──
        ax3 = axes[2]
        if self.market_tpi is not None:
            tpi = self.market_tpi[mask]
            ax3.plot(tpi.index, tpi.values, lw=1.5, color='#8338EC')
            ax3.axhline(0.5, ls='--', color='green', alpha=.5)
            ax3.axhline(0, ls='--', color='red', alpha=.5)
        ax3.set_title('Market TPI', fontsize=12)
        ax3.grid(True, alpha=.2, ls='--')

        plt.tight_layout()
        plt.show()

    # ─── Runner ────────────────────────────────────────────────

    def run(self, show_results=True, show_plot=True):
        self._fetch_data()
        self._align_data()
        self._compute_all_trends()
        self._run_tournament()
        self._find_top_performers()
        self._compute_all_alphas()
        self._find_highest_alpha()
        self._calc_raw_rsps_equity()
        self._calc_confidence_score()
        self._calc_market_tpi()
        self._calc_final_equity()
        if show_results:
            self._print_results()
        if show_plot:
            self._plot()
        return self


if __name__ == '__main__':
    eh = EventHorizon(
        timeframe='12h',
        start_date='2023-01-01',
    )
    eh.run()
