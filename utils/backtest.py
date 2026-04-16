#Backtesting

#    ╱|、
#  (˚ˎ 。7
#   |、˜〵
#   じしˍ,)ノ

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from xgboost import XGBClassifier

_DEFAULT_START   = '2024-01-01'
_DEFAULT_END     = str(datetime.now().date())
_DEFAULT_CAPITAL = 10_000
_DEFAULT_FEE     = 0.0005
_DEFAULT_SL      = 0.1


opt_start_date = '2023-01-01'   # Start of optimization / training
opt_end_date = '2023-12-31'     # End of optimization / training

bt_start_date = '2024-01-01'    # Start of out-of-sample backtest
bt_end_date = datetime.now().date()


class Backtester:
    """
    Crypto backtesting framework.

    mode : 'long_only' | 'long_short' | 'long_short_bar'
        long_only      — bar-by-bar equity, signals clipped to [0, 1]
        long_short     — trade-by-trade equity, signals in [-1, 0, 1]
        long_short_bar — bar-by-bar equity for long+short, P&L computed vs entry price

    strategy_type : 'classic' | 'ml'
        classic — strategy_func(df) → df with 'signal' column
        ml      — use ml_mode to choose variant

    ml_mode : 'static' | 'rolling'   (only when strategy_type='ml')
        static  — same interface as classic; eval_start_date defines OOS window
        rolling — walk-forward; strategy_func(df, opt_start, opt_end, bt_start, bt_end)
    """

    def __init__(self, exchange: str = 'binance', data_dir: str = './data'):
        self.exchange = getattr(ccxt, exchange)()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    # ─────────────────────────────── Data ────────────────────────────────

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.feather"

    def get_data(self, symbol: str, timeframe: str,
                 start_date: str = _DEFAULT_START,
                 end_date: str = _DEFAULT_END,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Return OHLCV data, using a local feather cache where possible."""
        cache_file = self._cache_path(symbol, timeframe)

        if cache_file.exists() and not force_refresh:
            df = pd.read_feather(cache_file).set_index('timestamp')
            if not df.empty and df.index[0] <= pd.Timestamp(start_date):
                fresh = (not end_date or
                         df.index[-1] >= pd.Timestamp(end_date) - pd.Timedelta(hours=6))
                if fresh:
                    df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    return df
                return self._download_with_cache(symbol, timeframe, start_date, end_date, cached_df=df)

        return self._download_with_cache(symbol, timeframe, start_date, end_date)

    def _download_with_cache(self, symbol: str, timeframe: str,
                              start_date: str, end_date: str,
                              cached_df: pd.DataFrame = None) -> pd.DataFrame:
        cache_file = self._cache_path(symbol, timeframe)
        since_date = pd.Timestamp(start_date)

        if cached_df is None and cache_file.exists():
            cached_df = pd.read_feather(cache_file).set_index('timestamp')

        if cached_df is not None and not cached_df.empty and cached_df.index[0] <= since_date:
            since_date = cached_df.index[-1]

        new_df = self._fetch_from_exchange(symbol, timeframe, since_date, start_date, end_date)

        if new_df.empty:
            if cached_df is not None and not cached_df.empty:
                result = cached_df[cached_df.index >= pd.Timestamp(start_date)]
                if end_date:
                    result = result[result.index <= pd.Timestamp(end_date)]
                return result
            return new_df

        if cached_df is not None and not cached_df.empty:
            cached_df = cached_df[cached_df.index < new_df.index[0]]
            combined  = pd.concat([cached_df, new_df])
            combined  = combined[~combined.index.duplicated(keep='last')]
        else:
            combined = new_df

        combined.reset_index().to_feather(cache_file)

        result = combined[combined.index >= pd.Timestamp(start_date)]
        if end_date:
            result = result[result.index <= pd.Timestamp(end_date)]
        return result

    def _fetch_from_exchange(self, symbol: str, timeframe: str,
                              since_date, start_date: str, end_date: str) -> pd.DataFrame:
        since         = int(since_date.timestamp() * 1000)
        all_ohlcv     = []
        current_since = since

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            except Exception as e:
                print(f'  ✗ {symbol}: {e}')
                break

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1

            if end_date and pd.to_datetime(ohlcv[-1][0], unit='ms') >= pd.Timestamp(end_date):
                break
            if len(ohlcv) < 1000:
                break

            time.sleep(self.exchange.rateLimit / 1000)

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol

        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df

    # ──────────────────────────── Equity ─────────────────────────────────

    def _calc_equity_bar_by_bar(self, df: pd.DataFrame, initial_capital: float,
                                 fee_per_trade: float, stop_loss: float):
        """Long-only: equity updated every bar for a smooth curve."""
        close  = df['close'].values
        low    = df['low'].values
        pos    = df['position'].values   # 0 or 1

        equity     = np.full(len(df), float(initial_capital))
        curr       = float(initial_capital)
        in_pos     = False
        entry_px   = 0.0
        trade_rets = []

        for i in range(1, len(df)):
            if in_pos:
                sl_px = entry_px * (1 - stop_loss)
                if stop_loss > 0 and low[i] <= sl_px:
                    exit_px = min(close[i], sl_px)
                    curr   *= exit_px / close[i - 1]
                    curr   *= (1 - fee_per_trade)
                    trade_rets.append(exit_px / entry_px - 1 - 2 * fee_per_trade)
                    in_pos  = False
                elif pos[i] == 0:
                    curr  *= close[i] / close[i - 1]
                    curr  *= (1 - fee_per_trade)
                    trade_rets.append(close[i] / entry_px - 1 - 2 * fee_per_trade)
                    in_pos = False
                else:
                    curr *= close[i] / close[i - 1]

            if not in_pos and pos[i] == 1 and pos[i - 1] != 1:
                entry_px = close[i]
                curr    *= (1 - fee_per_trade)
                in_pos   = True

            equity[i] = curr

        df['equity']   = equity
        df['drawdown'] = df['equity'] / df['equity'].cummax() - 1
        return df, trade_rets

    def _calc_equity_bar_by_bar_ls(self, df: pd.DataFrame, initial_capital: float,
                                    fee_per_trade: float, leverage: float):
        """Long+Short bar-by-bar: equity marked to market every bar using entry price.
        Leverage applied to returns; liquidation at entry_px ± 1/leverage wipes equity flat."""
        high      = df['high'].values
        low       = df['low'].values
        close     = df['close'].values
        positions = df['position'].values

        equity     = np.full(len(df), float(initial_capital))
        rets_arr   = np.zeros(len(df))
        curr       = float(initial_capital)
        trade_rets = []

        entry_px     = 0.0
        entry_equity = 0.0
        liq_px       = 0.0
        direction    = 0
        in_pos       = False
        liquidated   = False

        for i in range(1, len(df)):
            if liquidated:
                equity[i] = 0.0
                continue

            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            liq_hit  = False

            if in_pos and leverage > 1:
                if direction == 1 and low[i] <= liq_px:
                    liq_hit = True
                elif direction == -1 and high[i] >= liq_px:
                    liq_hit = True

            if in_pos and (liq_hit or prev_pos != curr_pos):
                if liq_hit:
                    curr        = 0.0
                    rets_arr[i] = -1.0
                    trade_rets.append(-1.0)
                    in_pos     = False
                    liquidated = True
                else:
                    ret         = direction * leverage * (close[i] - entry_px) / entry_px - 2 * fee_per_trade
                    curr        = entry_equity * (1 + ret)
                    rets_arr[i] = ret
                    trade_rets.append(ret)
                    in_pos    = False
                    direction = 0

            if not in_pos and not liquidated and curr_pos != 0 and prev_pos != curr_pos:
                entry_px     = close[i]
                liq_px       = (entry_px * (1 - 1 / leverage) if curr_pos == 1
                                else entry_px * (1 + 1 / leverage))
                curr        *= (1 - fee_per_trade)
                entry_equity = curr
                direction    = curr_pos
                in_pos       = True

            if in_pos:
                curr = entry_equity * (1 + direction * leverage * (close[i] - entry_px) / entry_px)

            equity[i] = curr

        df['strategy_returns'] = rets_arr
        df['equity']           = equity
        df['drawdown']         = df['equity'] / df['equity'].cummax() - 1
        return df, trade_rets

    def _calc_equity_trade_by_trade(self, df: pd.DataFrame, initial_capital: float,
                                     fee_per_trade: float, stop_loss: float):
        """Long+Short: equity updated only when a trade closes."""
        high      = df['high'].values
        low       = df['low'].values
        close     = df['close'].values
        positions = df['position'].values

        equity   = np.full(len(df), float(initial_capital))
        rets_arr = np.zeros(len(df))
        curr     = float(initial_capital)

        entry_px    = 0.0
        in_pos      = False
        direction   = 0
        actual_exit = 0.0

        for i in range(1, len(df)):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            stop_hit = False

            if in_pos and stop_loss > 0:
                if direction == 1 and low[i] <= entry_px * (1 - stop_loss):
                    stop_hit    = True
                    actual_exit = min(close[i], entry_px * (1 - stop_loss))
                elif direction == -1 and high[i] >= entry_px * (1 + stop_loss):
                    stop_hit    = True
                    actual_exit = max(close[i], entry_px * (1 + stop_loss))

            if in_pos and (stop_hit or prev_pos != curr_pos):
                exit_px     = actual_exit if stop_hit else close[i]
                ret         = ((exit_px - entry_px) / entry_px if direction == 1
                               else (entry_px - exit_px) / entry_px)
                ret        -= 2 * fee_per_trade
                curr       *= (1 + ret)
                rets_arr[i] = ret
                in_pos      = False
                direction   = 0

            if not in_pos and not stop_hit and curr_pos != 0 and prev_pos != curr_pos:
                entry_px  = close[i]
                direction = curr_pos
                in_pos    = True

            equity[i] = curr

        df['strategy_returns'] = rets_arr
        df['equity']           = equity
        df['drawdown']         = df['equity'] / df['equity'].cummax() - 1
        trade_rets             = df['strategy_returns'][df['strategy_returns'] != 0]
        return df, trade_rets

    def _calc_equity_long_short_sltp(self, df: pd.DataFrame,
                                      initial_capital: float,
                                      fee_per_trade: float):
        """
        Long+Short with fractional position sizing owned by the strategy.

        Expected signal semantics:
          - signal in [-1, 1] where sign = direction and |signal| = fraction
            of equity committed at entry (i.e. the position size).
          - Strategy manages its own SL/TP via the state-machine flipping
            signal back to 0 — no built-in stop_loss is applied here.
          - Equity is updated only on trade close; the realised return on
            equity for one trade is `size * (price_ret - 2*fee)`.

        Size is locked at entry (from |signal[entry_bar]|) and held until the
        strategy flips signal; intra-trade magnitude fluctuations are ignored.
        """
        close     = df['close'].values
        positions = df['position'].values

        equity   = np.full(len(df), float(initial_capital))
        rets_arr = np.zeros(len(df))
        curr     = float(initial_capital)

        entry_px  = 0.0
        entry_sz  = 0.0
        in_pos    = False
        direction = 0

        for i in range(1, len(df)):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]

            # Direction flip or flatten — close the trade at this bar's close
            if in_pos and (np.sign(curr_pos) != direction or curr_pos == 0):
                exit_px = close[i]
                price_ret = ((exit_px - entry_px) / entry_px if direction == 1
                             else (entry_px - exit_px) / entry_px)
                ret         = entry_sz * (price_ret - 2 * fee_per_trade)
                curr       *= (1 + ret)
                rets_arr[i] = ret
                in_pos      = False
                direction   = 0
                entry_sz    = 0.0

            # Open a new trade (or flipped side) at this bar's close
            if not in_pos and curr_pos != 0 and np.sign(curr_pos) != np.sign(prev_pos):
                entry_px  = close[i]
                direction = int(np.sign(curr_pos))
                entry_sz  = min(abs(curr_pos), 1.0)   # no leverage
                in_pos    = True

            equity[i] = curr

        df['strategy_returns'] = rets_arr
        df['equity']           = equity
        df['drawdown']         = df['equity'] / df['equity'].cummax() - 1
        trade_rets             = df['strategy_returns'][df['strategy_returns'] != 0]
        return df, trade_rets

    def _run_equity(self, df: pd.DataFrame, mode: str,
                    initial_capital: float, fee_per_trade: float,
                    stop_loss: float, leverage: float):
        """Clip signals and compute equity based on mode."""
        if mode == 'long_only':
            df['position'] = df['signal'].clip(0, 1)
            return self._calc_equity_bar_by_bar(df, initial_capital, fee_per_trade, stop_loss)
        if mode == 'long_short_bar':
            df['position'] = df['signal'].clip(-1, 1)
            return self._calc_equity_bar_by_bar_ls(df, initial_capital, fee_per_trade, leverage)
        if mode == 'long_short_sltp':
            df['position'] = df['signal'].clip(-1, 1)
            return self._calc_equity_long_short_sltp(df, initial_capital, fee_per_trade)
        df['position'] = df['signal'].clip(-1, 1)
        return self._calc_equity_trade_by_trade(df, initial_capital, fee_per_trade, stop_loss)

    # ──────────────────────────── Metrics ────────────────────────────────

    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(np.sqrt(len(returns)) * returns.mean() / returns.std())

    def _cagr(self, df: pd.DataFrame, total_return: float) -> float:
        days = (df.index[-1] - df.index[0]).days
        return 0.0 if days == 0 else (1 + total_return) ** (365 / days) - 1

    def _omega(self, returns: pd.Series) -> float:
        pos = returns[returns > 0].sum()
        neg = returns[returns < 0].sum()
        if neg == 0:
            return float('inf') if pos > 0 else float('nan')
        return float(pos / abs(neg))

    def _trade_metrics(self, trade_rets) -> Dict:
        if len(trade_rets) == 0:
            return {'ev': 0.0, 'win_rate': 0.0, 'max_lose_streak': 0}
        arr    = np.asarray(trade_rets, dtype=float)
        streak = mx = 0
        for r in arr:
            if r < 0:
                streak += 1
                mx      = max(mx, streak)
            else:
                streak = 0
        return {
            'ev':              float(arr.mean()),
            'win_rate':        float((arr > 0).mean()),
            'max_lose_streak': mx,
        }

    def _compute_metrics(self, eval_df: pd.DataFrame, trade_rets) -> Dict:
        tr        = pd.Series(trade_rets if isinstance(trade_rets, list)
                              else trade_rets.values, dtype=float)
        total_ret = eval_df['equity'].iloc[-1] / eval_df['equity'].iloc[0] - 1
        tm        = self._trade_metrics(tr)
        return {
            'total_return':    total_ret,
            'max_drawdown':    eval_df['drawdown'].min(),
            'cagr':            self._cagr(eval_df, total_ret),
            'sharpe':          self._sharpe(tr) if len(tr) > 0 else 0.0,
            'omega':           self._omega(tr)  if len(tr) > 0 else float('nan'),
            'trades':          len(tr),
            'ev':              tm['ev'],
            'win_rate':        tm['win_rate'],
            'max_lose_streak': tm['max_lose_streak'],
        }

    # ─────────────────────────── Display ─────────────────────────────────

    @staticmethod
    def _tag(mode: str, strategy_type: str, ml_mode: str) -> str:
        parts = [strategy_type.upper()]
        if strategy_type == 'ml':
            parts.append(ml_mode.upper())
        parts.append('LONG-ONLY' if mode == 'long_only' else 'LONG+SHORT')
        return '  ·  '.join(parts)

    def _print_header(self, start_date: str, end_date: str,
                      mode: str, strategy_type: str, ml_mode: str,
                      window_bars: int = None, step_bars: int = None):
        W   = 70
        tag = self._tag(mode, strategy_type, ml_mode)
        print(f'\n┌{"─"*(W-2)}┐')
        print(f'│  {"BACKTEST  ·  " + tag:<{W-4}}│')
        print(f'│  {f"{start_date}  →  {end_date}":<{W-4}}│')
        if strategy_type == 'ml' and ml_mode == 'rolling':
            print(f'│  {f"Window: {window_bars} bars  ·  Step: {step_bars} bars":<{W-4}}│')
        print(f'└{"─"*(W-2)}┘')

    @staticmethod
    def _row(l1: str, v1: str, l2: str = '', v2: str = '') -> str:
        left  = f'{l1:<13}{v1:<19}'
        right = f'{l2:<14}{v2:<18}' if l2 else ' ' * 32
        return f'│  {left}  {right}│'

    def _print_symbol_result(self, symbol: str, m: Dict,
                              mode: str, strategy_type: str, ml_mode: str):
        W        = 70
        tag      = self._tag(mode, strategy_type, ml_mode)
        sharpe_s = f"{m['sharpe']:.2f}" if np.isfinite(m['sharpe']) else 'N/A'
        omega_s  = f"{m['omega']:.2f}"  if np.isfinite(m['omega'])  else '∞'

        print(f'\n┌{"─"*(W-2)}┐')
        print(f'│  {f"{symbol}  ·  {tag}":<{W-4}}│')
        print(f'├{"─"*(W-2)}┤')
        print(self._row('Return',      f"{m['total_return']:+.2%}",
                         'Trades',      str(m['trades'])))
        print(self._row('CAGR',        f"{m['cagr']:+.2%}",
                         'Win Rate',    f"{m['win_rate']:.2%}"))
        print(self._row('Max DD',      f"{m['max_drawdown']:.2%}",
                         'Avg EV',      f"{m['ev']:+.2%}"))
        print(self._row('Sharpe',      sharpe_s,
                         'Omega',       omega_s))
        print(self._row('Lose Streak', str(m['max_lose_streak'])))
        print(f'└{"─"*(W-2)}┘')

    def _print_portfolio_summary(self, results: Dict,
                                  mode: str, strategy_type: str, ml_mode: str):
        W   = 70
        tag = self._tag(mode, strategy_type, ml_mode)
        n   = len(results)

        def avg(key):
            vals = [r[key] for r in results.values()
                    if np.isfinite(r.get(key, float('nan')))]
            return np.mean(vals) if vals else float('nan')

        avg_ev     = avg('ev')
        avg_sharpe = avg('sharpe')
        avg_omega  = avg('omega')
        avg_wr     = avg('win_rate')
        omega_s    = f'{avg_omega:.2f}' if np.isfinite(avg_omega) else '∞'
        title      = f'PORTFOLIO  ·  {n} symbol{"s" if n != 1 else ""}  ·  {tag}'

        print(f'\n┌{"─"*(W-2)}┐')
        print(f'│  {title:<{W-4}}│')
        print(f'├{"─"*(W-2)}┤')
        print(self._row('Avg EV',      f'{avg_ev:+.2%}',
                         'Avg Sharpe',  f'{avg_sharpe:.2f}'))
        print(self._row('Avg Omega',   omega_s,
                         'Avg WinRate', f'{avg_wr:.2%}'))
        print(f'└{"─"*(W-2)}┘')

    def _plot_equity_all(self, equity_map: Dict, label: str):
        """Plot all symbols' equity curves on a single chart."""
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (symbol, eval_df) in enumerate(equity_map.items()):
            if len(eval_df) < 2:
                continue
            norm = eval_df['equity'] / eval_df['equity'].iloc[0]
            ax.plot(eval_df.index, norm.values,
                    linewidth=2, color=colors[i % len(colors)], alpha=0.85,
                    label=symbol)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, which='both', alpha=0.2, linestyle='--')
        ax.grid(True, which='major', alpha=0.4)
        ax.set_title(f'{label}  ·  Normalised Log-Scale Equity',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity (normalised to 1)', fontsize=12)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

    # ─────────────────────── Internal runners ────────────────────────────

    def _process_classic(self, df: pd.DataFrame, strategy_func: Callable,
                          mode: str, initial_capital: float,
                          fee_per_trade: float, stop_loss: float,
                          leverage: float, eval_start_date: Optional[str]):
        result_df = strategy_func(df.copy())
        result_df, trade_rets = self._run_equity(
            result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
        eval_df = (result_df[result_df.index >= pd.Timestamp(eval_start_date)]
                   if eval_start_date else result_df)
        return result_df, eval_df, trade_rets, None

    def _process_rolling(self, df: pd.DataFrame, strategy_func: Callable,
                          mode: str, initial_capital: float,
                          fee_per_trade: float, stop_loss: float,
                          leverage: float, window_bars: int, step_bars: int):
        total_bars       = len(df)
        combined_signals = np.zeros(total_bars)
        total_steps      = (total_bars - window_bars + step_bars - 1) // step_bars
        n_steps          = 0
        pad              = len(str(total_steps))

        for i in range(window_bars, total_bars, step_bars):
            end_i       = min(i + step_bars, total_bars)
            train_start = df.index[i - window_bars]
            train_end   = df.index[i - 1]
            pred_start  = df.index[i]
            pred_end    = df.index[end_i - 1]
            n_steps    += 1

            print(f'  Step {n_steps:>{pad}}/{total_steps}  '
                  f'Train [{str(train_start)[:10]} → {str(train_end)[:10]}]  '
                  f'Predict [{str(pred_start)[:10]} → {str(pred_end)[:10]}]')

            res  = strategy_func(df.copy(),
                                 opt_start=train_start, opt_end=train_end,
                                 bt_start=pred_start,   bt_end=pred_end)
            mask = (df.index >= pred_start) & (df.index <= pred_end)
            combined_signals[mask] = res.loc[mask, 'signal'].values

        result_df           = df.copy()
        result_df['signal'] = combined_signals
        result_df, trade_rets = self._run_equity(
            result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
        eval_df = result_df[result_df.index >= df.index[window_bars]]
        return result_df, eval_df, trade_rets, n_steps

    # ───────────────────────── Public API ────────────────────────────────

    def backtest(self,
                 symbols:          List[str],
                 strategy_func:    Callable,
                 timeframe:        str,
                 start_date:       str           = _DEFAULT_START,
                 end_date:         str           = _DEFAULT_END,
                 mode:             str           = 'long_short',
                 strategy_type:    str           = 'classic',
                 ml_mode:          str           = 'static',
                 eval_start_date:  Optional[str] = None,
                 window_bars:      int           = 50000,
                 step_bars:        int           = 8000,
                 initial_capital:  float         = _DEFAULT_CAPITAL,
                 fee_per_trade:    float         = _DEFAULT_FEE,
                 stop_loss:        float         = _DEFAULT_SL,
                 leverage:         float         = 1.0,
                 show_plot:        bool          = True) -> Dict:
        """
        Run a backtest.

        Parameters
        ----------
        mode : 'long_only' | 'long_short' | 'long_short_bar'
        strategy_type : 'classic' | 'ml'
        ml_mode : 'static' | 'rolling'   (only when strategy_type='ml')
        eval_start_date : OOS start date for ml/static mode
        window_bars, step_bars : walk-forward parameters for ml/rolling mode
        """
        assert mode in ('long_only', 'long_short', 'long_short_bar', 'long_short_sltp'), \
            "mode must be 'long_only', 'long_short', 'long_short_bar', or 'long_short_sltp'"
        assert strategy_type in ('classic', 'ml'), \
            "strategy_type must be 'classic' or 'ml'"
        assert ml_mode in ('static', 'rolling'), \
            "ml_mode must be 'static' or 'rolling'"

        is_rolling = (strategy_type == 'ml' and ml_mode == 'rolling')

        self._print_header(start_date, end_date, mode, strategy_type, ml_mode,
                           window_bars if is_rolling else None,
                           step_bars   if is_rolling else None)

        print('\n  Loading data...')
        data_cache = {}
        for symbol in symbols:
            print(f'  → {symbol}')
            data_cache[symbol] = self.get_data(symbol, timeframe, start_date, end_date)

        results    = {}
        equity_map = {}
        for symbol in symbols:
            df = data_cache.get(symbol)
            if df is None or df.empty:
                print(f'\n  ✗ No data for {symbol}')
                continue

            if is_rolling and len(df) <= window_bars:
                print(f'\n  ✗ {symbol}: not enough data ({len(df)} bars < window {window_bars})')
                continue

            print(f'\n  Running {symbol}...')
            try:
                if is_rolling:
                    result_df, eval_df, trade_rets, n_steps = self._process_rolling(
                        df, strategy_func, mode, initial_capital,
                        fee_per_trade, stop_loss, leverage, window_bars, step_bars)
                else:
                    result_df, eval_df, trade_rets, n_steps = self._process_classic(
                        df, strategy_func, mode, initial_capital,
                        fee_per_trade, stop_loss, leverage, eval_start_date)

                if eval_df.empty or len(eval_df) < 2:
                    print(f'  ✗ No evaluation data for {symbol}')
                    continue

                m = self._compute_metrics(eval_df, trade_rets)
                results[symbol]    = {'data': result_df, **m}
                equity_map[symbol] = eval_df

                self._print_symbol_result(symbol, m, mode, strategy_type, ml_mode)

            except Exception as e:
                print(f'  ✗ Error processing {symbol}: {e}')
                raise

        if len(results) > 1:
            self._print_portfolio_summary(results, mode, strategy_type, ml_mode)

        if show_plot and equity_map:
            self._plot_equity_all(equity_map, self._tag(mode, strategy_type, ml_mode))

        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Strategies
# ═══════════════════════════════════════════════════════════════════════════

def sma_crossover_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 21) -> pd.DataFrame:
    """Simple Moving Average Crossover."""
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['signal']   = 0
    df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] =  1
    df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1
    return df




# ═══════════════════════════════════════════════════════════════════════════
#  Usage example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    bt = Backtester()
    


    results = bt.backtest(
        symbols        = ['SUI/USDT', 'ETH/USDT', 'XRP/USDT', 'DOT/USDT', 'ZEC/USDT', 'LTC/USDT', 'ADA/USDT', 'UNI/USDT', 'LINK/USDT', 'AVAX/USDT'],
        timeframe      = '1D',
        strategy_func  = sma_crossover_strategy,
        mode           = 'long_only', # long_short or long_only or long_short_bar or long_short_sltp
        strategy_type  = 'classic', # classic or ml
        ml_mode        = 'static', # rolling or static
        fee_per_trade  = 0.0005,
        stop_loss      = 0.5,
        leverage=2,
        start_date=opt_start_date,
        end_date=bt_end_date,
        eval_start_date=bt_start_date,
        
       
        
        
        
    )
