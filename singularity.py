#!/usr/bin/env python3
"""
SINGULARITY — Combined Portfolio System

Allocation at all times:
  Risk ON  → RISK_ON_ALLOC  × (EH_WEIGHT EH + GA_WEIGHT GA)
             + LG_ALLOC      × Lagrange
  Risk OFF → LG_ALLOC        × Lagrange  (rest is cash)

Defaults:
  Risk ON  → 80% (80% EH + 20% GA)  +  20% Lagrange
  Risk OFF → 20% Lagrange           +  80% cash

Allocation decisions are re-evaluated at the start of each rebalance period
(configurable, default = monthly).
"""

#    ╱|、
#  (˚ˎ 。7
#   |、˜〵
#   じしˍ,)ノ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime

from systems.EventHorizon import EventHorizon
import systems.gravityarena as gravityarena
import systems.lagrange as lagrange_mod
from utils.backtest import Backtester

# ─── Configuration ──────────────────────────────────────────────────────
RISK_ON_ALLOC    = 0.80    # Fraction of portfolio in EH+GA when risk-on
EH_WEIGHT        = 0.80    # EH share within the risk-on sleeve
GA_WEIGHT        = 0.20    # GA share within the risk-on sleeve
LG_ALLOC         = 0.20    # Fraction always allocated to Lagrange (both states)
# Risk ON  → RISK_ON_ALLOC × (EH_WEIGHT EH + GA_WEIGHT GA) + LG_ALLOC × LG
# Risk OFF → LG_ALLOC × LG  (rest is cash, return = 0)

REBALANCE_MONTHS = 1       # Re-evaluate allocation every N months

START_DATE   = '2023-01-01'
GA_STRATEGY  = 'Top 1'    # 'Top 1' or 'Top 4'

# Lagrange portfolio config (mirrors lagrange.py __main__ defaults)
LG_SYMBOLS = [
    'SUI/USDT', 'ETH/USDT', 'XRP/USDT', 'DOT/USDT', 'ZEC/USDT',
    'LTC/USDT', 'ADA/USDT', 'UNI/USDT', 'LINK/USDT', 'AVAX/USDT',
]
LG_COIN_PARAMS = {s: {
    'ema_small': 35, 'ema_big': 55, 'rsi_length': 35,
    'rsi_long_threshold': 70, 'rsi_short_threshold': 35,
    'left_len_high': 10, 'right_len_high': 5,
    'left_len_low': 5, 'right_len_low': 10,
    'er_period': 600, 'er_threshold': 5,
    'rs_length': 10, 'rs_lookback': 250,
    'garch_threshold_long': 50, 'garch_threshold_short': 30,
    'klass_threshold': 70,
} for s in LG_SYMBOLS}
LG_TIMEFRAME        = '5m'
LG_EXPOSURE         = 1.0
LG_STOP_LOSS        = 0.05
LG_INITIAL_CAPITAL  = 10_000


class Singularity:

    def __init__(self, start_date=START_DATE, end_date=None,
                 risk_on_alloc=RISK_ON_ALLOC,
                 eh_weight=EH_WEIGHT, ga_weight=GA_WEIGHT,
                 lg_alloc=LG_ALLOC,
                 ga_strategy=GA_STRATEGY,
                 rebalance_months=REBALANCE_MONTHS,
                 lg_symbols=None, lg_coin_params=None):
        self.start_date       = start_date
        self.end_date         = end_date or str(datetime.now().date())
        self.risk_on_alloc    = risk_on_alloc
        self.eh_weight        = eh_weight
        self.ga_weight        = ga_weight
        self.lg_alloc         = lg_alloc
        self.ga_strategy      = ga_strategy
        self.rebalance_months = rebalance_months
        self.lg_symbols       = lg_symbols or LG_SYMBOLS
        self.lg_coin_params   = lg_coin_params or LG_COIN_PARAMS
        self.bt               = Backtester()

    # ─── Run sub-systems ──────────────────────────────────────────

    def _run_event_horizon(self):
        print('\n' + '=' * 70)
        print('  [1/3]  EventHorizon')
        print('=' * 70)
        eh = EventHorizon(
            timeframe='1d',
            start_date=self.start_date,
            end_date=self.end_date,
        )
        eh.run(show_results=False, show_plot=False)
        return eh

    def _run_gravity_arena(self):
        print('\n' + '=' * 70)
        print('  [2/3]  GravityArena')
        print('=' * 70)
        warmup = str((pd.Timestamp(self.start_date) - pd.Timedelta(days=200)).date())
        gravityarena.start_date    = warmup
        gravityarena.BACKTEST_START = self.start_date
        gravityarena.end_date      = self.end_date
        return gravityarena.run_backtest(self.bt, show_results=False, show_plot=False)

    def _run_lagrange(self):
        print('\n' + '=' * 70)
        print('  [3/3]  Lagrange')
        print('=' * 70)
        # Run Lagrange backtest directly
        eq_df = lagrange_mod.run_backtest(
            bt=self.bt,
            show_results=False,
            show_plot=False
        )
        # Return normalized daily equity
        eq = eq_df['Hedge (L+S)']
        return eq / eq.iloc[0]

    # ─── Combine ──────────────────────────────────────────────────

    def _align_and_combine(self, eh, ga_eq, lg_daily):
        start_ts = pd.Timestamp(self.start_date)

        # EventHorizon equity & running flag (daily)
        eh_eq  = eh.final_equity[eh.final_equity.index >= start_ts]
        eh_run = eh.rsps_running[eh.rsps_running.index >= start_ts]

        # GravityArena — may be 8h, resample to daily (last value per day)
        ga_series = ga_eq[self.ga_strategy].copy()
        ga_series.index = ga_series.index.normalize()
        ga_daily = ga_series.groupby(ga_series.index).last()

        # Common daily index
        common = (eh_eq.index
                  .intersection(ga_daily.index)
                  .intersection(lg_daily.index))
        eh_eq   = eh_eq.reindex(common)
        eh_run  = eh_run.reindex(common).fillna(False)
        ga_daily = ga_daily.reindex(common)
        lg_daily = lg_daily.reindex(common)

        # Daily returns
        eh_ret = eh_eq.pct_change().fillna(0)
        ga_ret = ga_daily.pct_change().fillna(0)
        lg_ret = lg_daily.pct_change().fillna(0)

        # ── Monthly rebalancing ──────────────────────────────────
        # Allocation decision is locked in at the start of each rebalance
        # window and held for the full period.
        n          = len(common)
        combined   = np.ones(n)
        mode       = np.empty(n, dtype=object)
        mode[:]    = 'OFF'

        # Determine which "slot" each date belongs to.
        # A new slot starts whenever floor(month_offset / rebalance_months) changes.
        def _slot(ts):
            total_months = ts.year * 12 + ts.month
            return total_months // self.rebalance_months

        current_mode  = 'OFF'
        current_slot  = None

        for i in range(1, n):
            date = common[i]
            slot = _slot(date)

            # Re-evaluate at the start of a new rebalance window
            if slot != current_slot:
                current_slot = slot
                # Use EH signal from previous bar (no lookahead)
                current_mode = 'ON' if bool(eh_run.iloc[i - 1]) else 'OFF'

            mode[i] = current_mode
            if current_mode == 'ON':
                # 80% into (80% EH + 20% GA), 20% into Lagrange
                ehga_ret = (self.eh_weight * eh_ret.iloc[i]
                            + self.ga_weight * ga_ret.iloc[i])
                port_ret = self.risk_on_alloc * ehga_ret + self.lg_alloc * lg_ret.iloc[i]
            else:
                # 20% into Lagrange, 80% cash (return = 0)
                port_ret = self.lg_alloc * lg_ret.iloc[i]

            combined[i] = combined[i - 1] * (1 + port_ret)

        self.combined_equity = pd.Series(combined, index=common)
        self.mode            = pd.Series(mode,     index=common)
        self.eh_equity       = eh_eq
        self.ga_equity       = ga_daily
        self.lg_equity       = lg_daily
        self.eh              = eh

    # ─── Metrics ──────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(equity):
        eq = equity.dropna()
        if len(eq) < 2:
            return None
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        daily_ret = eq.pct_change().dropna()
        days = (eq.index[-1] - eq.index[0]).days
        cagr = (1 + total_ret) ** (365 / max(days, 1)) - 1 if days > 0 else 0

        dd     = eq / eq.cummax() - 1
        max_dd = dd.min()

        std    = daily_ret.std()
        sharpe = np.sqrt(365) * daily_ret.mean() / std if std > 0 else 0.0

        neg_std = daily_ret[daily_ret < 0].std()
        sortino = np.sqrt(365) * daily_ret.mean() / neg_std if neg_std > 0 else 0.0

        pos   = daily_ret[daily_ret > 0].sum()
        neg   = daily_ret[daily_ret < 0].sum()
        omega = pos / abs(neg) if neg != 0 else float('inf')

        return {
            'total_return': total_ret, 'cagr': cagr,
            'max_drawdown': max_dd,    'sharpe': sharpe,
            'sortino': sortino,        'omega': omega,
        }

    # ─── Display ──────────────────────────────────────────────────

    def _print_results(self):
        W = 70
        print(f'\n{"=" * W}')
        print(f'  SINGULARITY')
        print(f'  {self.start_date}  ->  {self.end_date}')
        eh_eff = self.risk_on_alloc * self.eh_weight
        ga_eff = self.risk_on_alloc * self.ga_weight
        print(f'  Risk ON:   {eh_eff:.0%} EH  +  {ga_eff:.0%} GA  +  {self.lg_alloc:.0%} Lagrange')
        print(f'  Risk OFF:  {self.lg_alloc:.0%} Lagrange  +  {1-self.lg_alloc:.0%} cash')
        print(f'  Rebalance: every {self.rebalance_months} month(s)')
        print(f'{"=" * W}')

        last_mode    = self.mode.iloc[-1]
        risk_on_pct  = (self.mode == 'ON').sum() / len(self.mode) * 100
        print(f'\n  Current Mode:    {"RISK ON" if last_mode == "ON" else "RISK OFF"}')
        print(f'  Time Risk ON:    {risk_on_pct:.1f}%')
        print(f'  Time Risk OFF:   {100 - risk_on_pct:.1f}%')

        last = -1
        held = self.eh.held_asset.iloc[last]
        top  = self.eh.top_asset.iloc[last]
        conf = 'High' if self.eh.confidence.iloc[last] == 1 else 'Low'
        print(f'\n  EH Holding:      {held}')
        print(f'  EH RSPS Top:     {top}')
        print(f'  EH Confidence:   {conf}')

        header = (f'  {"Strategy":<26} {"Return":>9} {"CAGR":>9}'
                  f' {"Max DD":>9} {"Sharpe":>8} {"Sortino":>9} {"Omega":>8}')
        print(f'\n{header}')
        print(f'  {"-" * (len(header) - 2)}')

        for label, eq in [
            ('Singularity (Combined)', self.combined_equity),
            ('EventHorizon',           self.eh_equity),
            ('GravityArena',           self.ga_equity),
            ('Lagrange',               self.lg_equity),
        ]:
            m = self._compute_metrics(eq)
            if not m:
                continue
            print(f'  {label:<26} {m["total_return"]:>+8.2%} {m["cagr"]:>+8.2%}'
                  f' {m["max_drawdown"]:>8.2%} {m["sharpe"]:>8.2f}'
                  f' {m["sortino"]:>9.2f} {m["omega"]:>8.2f}')

    def _plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                 height_ratios=[3, 1],
                                 gridspec_kw={'hspace': 0.3})

        def norm(s):
            return s / s.iloc[0]

        comb = norm(self.combined_equity)
        eh   = norm(self.eh_equity)
        ga   = norm(self.ga_equity)
        lg   = norm(self.lg_equity)

        ax = axes[0]
        ax.plot(comb.index, comb.values, lw=2.5, color='#FFD700', label='Singularity')
        ax.plot(eh.index,   eh.values,   lw=1.2, color='#FF006E', alpha=0.7, label='EventHorizon')
        ax.plot(ga.index,   ga.values,   lw=1.2, color='#3A86FF', alpha=0.7, label='GravityArena')
        ax.plot(lg.index,   lg.values,   lw=1.2, color='#8338EC', alpha=0.7, label='Lagrange')

        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, alpha=.2, ls='--')
        ax.set_title(
            f'Singularity · Combined Portfolio'
            f'  (rebalance every {self.rebalance_months}M)',
            fontsize=16, fontweight='bold',
        )
        ax.legend(fontsize=10)

        ax2 = axes[1]
        risk_on = (self.mode == 'ON').astype(int)
        ax2.fill_between(risk_on.index, 0, 1, where=risk_on == 1,
                         alpha=0.3, color='green', label='Risk ON')
        ax2.fill_between(risk_on.index, 0, 1, where=risk_on == 0,
                         alpha=0.3, color='red',   label='Risk OFF')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['OFF', 'ON'])
        ax2.set_title('Regime', fontsize=12)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=.2, ls='--')

        plt.tight_layout()
        plt.show()

    # ─── Runner ───────────────────────────────────────────────────

    def run(self):
        eh      = self._run_event_horizon()
        ga_eq   = self._run_gravity_arena()
        lg_eq   = self._run_lagrange()
        self._align_and_combine(eh, ga_eq, lg_eq)
        self._print_results()
        self._plot()
        return self


if __name__ == '__main__':
    s = Singularity(
        start_date='2023-01-01',
        risk_on_alloc=0.60,   # 80% into EH+GA when risk-on
        eh_weight=0.80,       # EH share within risk-on sleeve
        ga_weight=0.20,       # GA share within risk-on sleeve
        lg_alloc=0.60,        # Lagrange allocation at all times
        rebalance_months=1,   # monthly rebalancing
    )
    s.run()
