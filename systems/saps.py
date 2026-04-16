# Strategic Adaptive Portfolio System (SAPS)
#
#    ╱|、
#  (˚ˎ 。7
#   |、˜〵
#   じしˍ,)ノ
#
# Long/Short crypto portfolio using genesis_v1 with adaptive weighting.
# Bar-by-bar mark-to-market equity with 4 weighting schemes.

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


from strategies.quark import quark
from utils.backtest import Backtester

_DEFAULT_FEE = 0.005
_DEFAULT_SL = 1

# ═══════════════════════════════════════════════════════════════════════════
#  CHOOSE YOUR STRATEGY HERE
# ═══════════════════════════════════════════════════════════════════════════
strategy = quark  # or quark, or any other strategy function


# Minimum raw score before normalising — guarantees every asset keeps
# a non-zero allocation.
_SCORE_FLOOR = 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Weighting functions  (operate on per-bar strategy returns)
# ═══════════════════════════════════════════════════════════════════════════

def _weight_sharpe(returns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """asset_sharpe / sum_of_all_sharpes."""
    scores = {}
    for sym, rets in returns.items():
        if len(rets) < 2 or np.std(rets) == 0:
            scores[sym] = _SCORE_FLOOR
        else:
            scores[sym] = max(_SCORE_FLOOR,
                              np.mean(rets) / np.std(rets) * np.sqrt(len(rets)))
    return _normalise(scores)


def _weight_sortino(returns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """asset_sortino / sum_of_all_sortinos."""
    scores = {}
    for sym, rets in returns.items():
        down = rets[rets < 0]
        if len(down) < 1 or np.std(down) == 0:
            scores[sym] = max(_SCORE_FLOOR,
                              np.mean(rets) * 10 if len(rets) > 0 else _SCORE_FLOOR)
        else:
            scores[sym] = max(_SCORE_FLOOR,
                              np.mean(rets) / np.std(down) * np.sqrt(len(rets)))
    return _normalise(scores)


def _weight_omega(returns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """asset_omega / sum_of_all_omegas."""
    scores = {}
    for sym, rets in returns.items():
        pos = rets[rets > 0].sum()
        neg = abs(rets[rets < 0].sum())
        if neg == 0:
            scores[sym] = max(_SCORE_FLOOR, pos * 10)
        else:
            scores[sym] = max(_SCORE_FLOOR, pos / neg)
    return _normalise(scores)


def _weight_kelly(returns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Half-Kelly fraction per asset, normalised."""
    scores = {}
    for sym, rets in returns.items():
        if len(rets) < 2:
            scores[sym] = _SCORE_FLOOR
            continue
        wins   = rets[rets > 0]
        losses = rets[rets < 0]
        wr     = len(wins) / len(rets)
        avg_w  = np.mean(wins)   if len(wins)   > 0 else 0.0
        avg_l  = abs(np.mean(losses)) if len(losses) > 0 else 1e-9
        if avg_w == 0:
            scores[sym] = _SCORE_FLOOR
        else:
            kelly = wr - (1 - wr) * (avg_l / avg_w)
            scores[sym] = max(_SCORE_FLOOR, kelly * 0.5)
    return _normalise(scores)


def _normalise(scores: Dict[str, float], max_weight: float = 1.0) -> Dict[str, float]:
    """Normalise to sum = 1.  If any weight exceeds *max_weight*, it is
    clamped and the excess is redistributed proportionally among the
    remaining assets.  Floor guarantees no asset reaches 0 %."""
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {s: 1.0 / n for s in scores} if n > 0 else {}
    weights = {s: v / total for s, v in scores.items()}
    # iterative clamping — keeps redistributing until nothing exceeds cap
    for _ in range(len(weights)):
        clamped = {s: w for s, w in weights.items() if w >= max_weight}
        if not clamped:
            break
        free = {s: w for s, w in weights.items() if w < max_weight}
        free_total = sum(free.values())
        for s in clamped:
            weights[s] = max_weight
        surplus = sum(w - max_weight for w in clamped.values())
        if free_total > 0 and surplus > 0:
            for s in free:
                weights[s] += surplus * (free[s] / free_total)
    return weights


WEIGHT_FUNCS = {
    'sharpe':  _weight_sharpe,
    'sortino': _weight_sortino,
    'omega':   _weight_omega,
    'kelly':   _weight_kelly,
}


# ═══════════════════════════════════════════════════════════════════════════
#  SAPS
# ═══════════════════════════════════════════════════════════════════════════

class SAPS:
    """
    Strategic Adaptive Portfolio System.

    Runs a strategy (configured at line 26: `strategy = genesis_v1` or `quark`)
    on multiple crypto assets with per-coin parameter overrides, then allocates
    capital using adaptive weights. Equity is computed bar-by-bar (mark-to-market).

    Parameters
    ----------
    symbols : list of str
        Trading pairs, e.g. ['BTC/USDT', 'ETH/USDT'].
    timeframe : str
        OHLCV timeframe, e.g. '4h', '1d'.
    weighting : str
        One of 'sharpe', 'sortino', 'omega', 'kelly'.
    coin_params : dict, optional
        Per-coin strategy parameter overrides.
        Example: {'BTC/USDT': {'mad_len': 30, 'tpi_long': 0.4}}
    start_date, end_date : str
        Backtest date range.
    initial_capital : float
        Starting equity.
    fee_per_trade : float
        Fee per side (entry or exit).
    stop_loss : float
        Stop-loss distance as a fraction (0.10 = 10%, 1.0 = off).
    rebalance_bars : int
        How often to recompute weights (in bars).
    weight_lookback : int
        Number of past bars of *strategy returns* used to compute
        weights. 0 = use the full history up to the current bar.
    max_weight : float
        Maximum weight any single asset can receive (e.g. 0.30 = 30 %).
        Excess is redistributed proportionally. 1.0 = no cap.
    """

    def __init__(
        self,
        symbols:          List[str],
        timeframe:        str          = '4h',
        weighting:        str          = 'sharpe',
        coin_params:      Optional[Dict[str, Dict]] = None,
        start_date:       str          = '2023-01-01',
        end_date:         str          = str(datetime.now().date()),
        initial_capital:  float        = 10_000,
        fee_per_trade:    float        = _DEFAULT_FEE,
        stop_loss:        float        = _DEFAULT_SL,
        rebalance_bars:   int          = 42,
        weight_lookback:  int          = 90,    # 0 = full history
        max_weight:       float        = 1.0,   # 1.0 = no cap
        exchange:         str          = 'binance',
        data_dir:         str          = './data',
    ):
        assert weighting in WEIGHT_FUNCS, (
            f"weighting must be one of {list(WEIGHT_FUNCS)}, got '{weighting}'"
        )
        self.symbols         = symbols
        self.timeframe       = timeframe
        self.weighting       = weighting
        self.coin_params     = coin_params or {}
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_capital = initial_capital
        self.fee_per_trade   = fee_per_trade
        self.stop_loss       = stop_loss
        self.rebalance_bars  = rebalance_bars
        self.weight_lookback = weight_lookback
        self.max_weight      = max_weight
        self.bt              = Backtester(exchange, data_dir)

    # ────────────────────── data + signals ─────────────────────────────

    def _fetch_and_signal(self) -> Dict[str, pd.DataFrame]:
        """Download data and run strategy per coin."""
        results = {}
        for symbol in self.symbols:
            print(f'  Fetching {symbol}...')
            df = self.bt.get_data(symbol, self.timeframe,
                                  self.start_date, self.end_date)
            if df is None or df.empty:
                print(f'  ✗ No data for {symbol}')
                continue
            params = self.coin_params.get(symbol, {})
            print(f'  Running {strategy.__name__} on {symbol}...')
            df = strategy(df.copy(), **params)
            df['position'] = df['signal']
            results[symbol] = df
        return results

    # ────────────────────── bar returns & weighting ───────────────────

    @staticmethod
    def _bar_returns(df: pd.DataFrame) -> np.ndarray:
        """Per-bar strategy returns: position * pct_change.
        Long bars get positive return when price rises,
        short bars get positive return when price falls."""
        close = df['close'].values
        pos   = df['position'].values
        rets  = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i - 1] != 0:
                rets[i] = pos[i] * (close[i] - close[i - 1]) / close[i - 1]
        return rets

    def _compute_weights(
        self,
        bar_rets: Dict[str, np.ndarray],
        current_bar: int,
    ) -> Dict[str, float]:
        """Compute weights from rolling bar returns."""
        if self.weight_lookback > 0:
            start = max(0, current_bar - self.weight_lookback)
        else:
            start = 0
        window = {sym: rets[start:current_bar]
                  for sym, rets in bar_rets.items()}
        raw = WEIGHT_FUNCS[self.weighting](window)
        return _normalise(
            {s: v for s, v in raw.items()},   # already normalised, but
            max_weight=self.max_weight,        # re-normalise with cap
        )

    # ────────────────────── portfolio simulation ──────────────────────

    def run(self, show_plot: bool = True) -> Dict:
        """Execute the full SAPS backtest (bar-by-bar mark-to-market)."""
        lookback_label = (f'{self.weight_lookback} bars'
                          if self.weight_lookback > 0 else 'full history')
        W = 70
        print(f'\n┌{"─" * (W - 2)}┐')
        print(f'│  {"SAPS  ·  Strategic Adaptive Portfolio System":<{W - 4}}│')
        print(f'│  {f"Weighting: {self.weighting.upper()}  ·  {len(self.symbols)} assets":<{W - 4}}│')
        print(f'│  {f"{self.start_date}  →  {self.end_date}":<{W - 4}}│')
        print(f'│  {f"Rebalance: {self.rebalance_bars} bars  ·  Lookback: {lookback_label}":<{W - 4}}│')
        print(f'└{"─" * (W - 2)}┘')

        # 1. fetch data + run strategy
        print('\n  Loading data & generating signals...')
        strategy_data = self._fetch_and_signal()
        if not strategy_data:
            print('  ✗ No data for any symbol.')
            return {}

        active_symbols = list(strategy_data.keys())
        n_assets       = len(active_symbols)

        # 2. build unified time index & align
        all_idx = sorted(set().union(*(df.index for df in strategy_data.values())))
        aligned = {sym: df.reindex(all_idx, method='ffill')
                   for sym, df in strategy_data.items()}
        n_bars  = len(all_idx)

        # 3. precompute per-bar strategy returns on the aligned index
        bar_rets = {}
        for sym in active_symbols:
            bar_rets[sym] = self._bar_returns(aligned[sym])

        # 4. bar-by-bar portfolio simulation
        portfolio_equity = np.full(n_bars, float(self.initial_capital))
        cash             = float(self.initial_capital)

        # active positions: sym -> {direction, entry_price, capital}
        # capital = amount allocated at entry (after entry fee)
        active           = {}
        all_trade_rets   = []
        per_coin_trades  = {s: [] for s in active_symbols}

        # weights — start equal
        current_weights  = {s: 1.0 / n_assets for s in active_symbols}
        weight_history   = []
        next_rebalance   = self.rebalance_bars

        for i in range(1, n_bars):
            # ── rebalance weights ──
            if i >= next_rebalance:
                current_weights = self._compute_weights(bar_rets, i)
                weight_history.append((i, dict(current_weights)))
                next_rebalance = i + self.rebalance_bars

            # ── process each asset ──
            exits_to_process  = []
            entries_to_process = []

            for sym in active_symbols:
                df       = aligned[sym]
                prev_pos = df['position'].iloc[i - 1]
                curr_pos = df['position'].iloc[i]
                if pd.isna(prev_pos): prev_pos = 0
                if pd.isna(curr_pos): curr_pos = 0
                close_px = df['close'].iloc[i]
                if pd.isna(close_px):
                    continue
                high_px = df['high'].iloc[i]
                low_px  = df['low'].iloc[i]

                # stop-loss check
                stop_hit = False
                exit_px  = close_px
                if sym in active and self.stop_loss < 1.0:
                    p = active[sym]
                    if p['direction'] == 1:
                        sl = p['entry_price'] * (1 - self.stop_loss)
                        if low_px <= sl:
                            stop_hit = True
                            exit_px  = min(close_px, sl)
                    else:
                        sl = p['entry_price'] * (1 + self.stop_loss)
                        if high_px >= sl:
                            stop_hit = True
                            exit_px  = max(close_px, sl)

                signal_changed = (prev_pos != curr_pos)

                if stop_hit or signal_changed:
                    if sym in active:
                        exits_to_process.append((sym, exit_px if stop_hit else close_px, stop_hit))
                    if not stop_hit and curr_pos != 0 and signal_changed:
                        entries_to_process.append((sym, int(curr_pos), close_px))
                    elif stop_hit:
                        pass  # don't re-enter on the bar we got stopped

            # ── execute exits ──
            for sym, ex_px, was_stop in exits_to_process:
                p   = active[sym]
                d   = p['direction']
                ret = d * (ex_px - p['entry_price']) / p['entry_price']
                # position value at exit
                pos_value = p['capital'] * (1 + ret)
                pos_value *= (1 - self.fee_per_trade)   # exit fee
                cash += pos_value
                # record trade return (entry fee already deducted from capital)
                trade_ret = ret - 2 * self.fee_per_trade
                all_trade_rets.append(trade_ret)
                per_coin_trades[sym].append(trade_ret)
                del active[sym]

            # ── mark-to-market before entries (for position sizing) ──
            mtm = 0.0
            for sym, p in active.items():
                c = aligned[sym]['close'].iloc[i]
                if pd.isna(c):
                    continue
                mtm += p['capital'] * (1 + p['direction'] * (c - p['entry_price']) / p['entry_price'])
            equity_now = cash + mtm

            # ── execute entries ──
            for sym, direction, entry_px in entries_to_process:
                w     = current_weights.get(sym, 1.0 / n_assets)
                alloc = equity_now * w
                if alloc <= 0:
                    continue
                cash -= alloc
                capital_after_fee = alloc * (1 - self.fee_per_trade)
                active[sym] = {
                    'direction':   direction,
                    'entry_price': entry_px,
                    'capital':     capital_after_fee,
                }

            # ── mark-to-market for equity recording ──
            mtm = 0.0
            for sym, p in active.items():
                c = aligned[sym]['close'].iloc[i]
                if pd.isna(c):
                    continue
                mtm += p['capital'] * (1 + p['direction'] * (c - p['entry_price']) / p['entry_price'])
            portfolio_equity[i] = cash + mtm

        # 5. results dataframe
        portfolio_df = pd.DataFrame({'equity': portfolio_equity}, index=all_idx)
        portfolio_df['drawdown'] = portfolio_df['equity'] / portfolio_df['equity'].cummax() - 1

        total_return = portfolio_df['equity'].iloc[-1] / self.initial_capital - 1
        max_dd       = portfolio_df['drawdown'].min()
        days         = (all_idx[-1] - all_idx[0]).days
        cagr         = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0

        n_trades = len(all_trade_rets)

        # portfolio-level metrics from equity bar returns
        eq = portfolio_df['equity'].values
        bar_pct = np.diff(eq) / eq[:-1]          # bar-by-bar % change
        bar_pct = bar_pct[np.isfinite(bar_pct)]

        if len(bar_pct) > 1 and np.std(bar_pct) > 0:
            sharpe  = float(np.sqrt(365) * np.mean(bar_pct) / np.std(bar_pct))
            down    = bar_pct[bar_pct < 0]
            sortino = (float(np.sqrt(365) * np.mean(bar_pct) / np.std(down))
                       if len(down) > 0 and np.std(down) > 0 else 0.0)
            pos_sum = bar_pct[bar_pct > 0].sum()
            neg_sum = bar_pct[bar_pct < 0].sum()
            omega   = float(pos_sum / abs(neg_sum)) if neg_sum != 0 else np.inf
        else:
            sharpe = sortino = omega = 0.0

        # 6. print
        self._print_coin_breakdown(per_coin_trades)
        self._print_results(total_return, max_dd, cagr, n_trades,
                            sharpe, omega, sortino)
        self._print_final_weights(current_weights)

        # 7. plot
        if show_plot and n_bars > 1:
            self._plot(portfolio_df, total_return, max_dd, weight_history,
                       active_symbols)

        return {
            'portfolio_df':     portfolio_df,
            'total_return':     total_return,
            'max_drawdown':     max_dd,
            'cagr':             cagr,
            'sharpe':           sharpe,
            'omega':            omega,
            'sortino':          sortino,
            'total_trades':     n_trades,
            'per_coin_trades':  per_coin_trades,
            'weight_history':   weight_history,
            'strategy_data':    strategy_data,
        }

    # ────────────────────── printing ──────────────────────────────────

    def _print_coin_breakdown(self, per_coin_trades: Dict[str, list]):
        print('\n' + '=' * 60)
        print('  PER-COIN BREAKDOWN')
        print('=' * 60)
        for sym in self.symbols:
            trades = per_coin_trades.get(sym, [])
            if trades:
                print(f'  {sym:14s} | Trades: {len(trades):4d}')
            else:
                print(f'  {sym:14s} | Trades:    0')

    def _print_results(self, total_return, max_dd, cagr, n_trades,
                       sharpe, omega, sortino):
        W = 60
        print('\n' + '=' * W)
        print('  SAPS PORTFOLIO RESULTS')
        print('=' * W)
        print(f'  Return: {total_return:+.2%}  |  Max DD: {max_dd:.2%}  |  CAGR: {cagr:+.2%}')
        print(f'  Trades: {n_trades}  |  Sharpe: {sharpe:.2f}  |  '
              f'Omega: {omega:.2f}  |  Sortino: {sortino:.2f}')
        print(f'  Weighting: {self.weighting.upper()}  |  '
              f'Fee: {self.fee_per_trade:.4%}/side  |  SL: {self.stop_loss:.2%}')
        print('=' * W)

    def _print_final_weights(self, weights: Dict[str, float]):
        print('\n  Final portfolio weights:')
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = '█' * int(w * 40)
            print(f'    {sym:14s}  {w:6.1%}  {bar}')

    # ────────────────────── plotting ──────────────────────────────────

    def _plot(self, portfolio_df: pd.DataFrame, total_return: float,
              max_dd: float, weight_history: list, symbols: list):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})

        ax = axes[0]
        ax.plot(portfolio_df.index, portfolio_df['equity'].values,
                linewidth=2, color='#2962FF', alpha=0.85)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, which='both', alpha=0.2, linestyle='--')
        ax.grid(True, which='major', alpha=0.4)
        ax.set_title(f'SAPS  ·  {self.weighting.upper()} weighting  ·  '
                     f'{len(symbols)} assets',
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Equity (log $)', fontsize=12)

        final_eq = portfolio_df['equity'].iloc[-1]
        ax.text(0.02, 0.98,
                f'Final: ${final_eq:,.0f}\nReturn: {total_return:+.1%}\n'
                f'Max DD: {max_dd:.1%}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2 = axes[1]
        if weight_history:
            bars   = [portfolio_df.index[wh[0]] for wh in weight_history]
            w_data = {s: [wh[1].get(s, 0.0) for wh in weight_history]
                      for s in symbols}
            bottom = np.zeros(len(bars))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for idx, sym in enumerate(symbols):
                vals = np.array(w_data[sym])
                ax2.fill_between(bars, bottom, bottom + vals,
                                 alpha=0.7, color=colors[idx % len(colors)],
                                 label=sym)
                bottom += vals
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Weight', fontsize=11)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.legend(fontsize=9, ncol=min(len(symbols), 5),
                       loc='upper center', bbox_to_anchor=(0.5, -0.15))
            ax2.grid(True, alpha=0.2, linestyle='--')
            ax2.set_title('Weight Evolution', fontsize=12, pad=8)

        plt.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    coins = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT',
        'BNB/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'XLM/USDT', 'XMR/USDT',
    ]

    saps = SAPS(
        symbols=coins,
        timeframe='1d',
        weighting='omega',          # 'sharpe' | 'sortino' | 'omega' | 'kelly',
        start_date='2023-01-01',
        initial_capital=10_000,
        fee_per_trade=0.005,
        stop_loss=1,
        rebalance_bars=1,           # rebalance every 42 bars
        weight_lookback=30,          # rolling window for metrics (0 = full history)
        max_weight=0.125,             # cap any single asset at 30 % (1.0 = no cap)
    )

    results = saps.run()
