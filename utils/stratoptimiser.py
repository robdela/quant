# Strategy Optimiser
#
#    ╱|、
#  (˚ˎ 。7
#   |、˜〵
#   じしˍ,)ノ

from utils.backtest import genesis_v1
import gc
import inspect
import numpy as np
import os
import concurrent.futures
from typing import Dict, List, Callable, Tuple
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time as _time
from datetime import datetime

from utils.backtest import (
    Backtester,
    _DEFAULT_START, _DEFAULT_END,
    _DEFAULT_CAPITAL, _DEFAULT_FEE, _DEFAULT_SL,
)


def _worker_evaluate(strategy_func, data_cache, params, symbols,
                     mode, initial_capital, fee_per_trade, stop_loss,
                     leverage, metric):
    """Module-level worker for parallel parameter evaluation (must be picklable)."""
    bt = object.__new__(Backtester)  # lightweight instance, no exchange needed
    values = []
    for symbol in symbols:
        df_base = data_cache.get(symbol)
        if df_base is None or df_base.empty:
            continue
        try:
            result_df = strategy_func(df_base.copy(), **params)
            result_df, trade_rets = bt._run_equity(
                result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
            metrics = StrategyOptimiser._enrich_metrics(
                bt._compute_metrics(result_df, trade_rets), trade_rets)
            v = metrics.get(metric, float('nan'))
            if np.isfinite(v):
                values.append(v)
            del result_df, trade_rets
        except Exception:
            pass
    return np.mean(values) if values else float('-inf')


class StrategyOptimiser:
    """
    Stepwise optimiser for classic strategies.

    Instead of an exhaustive grid search, parameters are optimised
    one at a time while the others are held fixed.  This dramatically
    reduces the number of backtests and limits overfitting.

    Process
    -------
    1. Run the strategy with default parameters (first value of each range).
    2. Sweep the first parameter across all its choices, pick the best.
    3. Lock that value and sweep the second parameter, pick the best.
    4. Repeat until every parameter has been optimised once (= 1 round).
    5. Optionally repeat for ``n_rounds`` rounds, always starting from
       the best values found in the previous round.

    mode : 'long_only' | 'long_short'
        long_only  — bar-by-bar equity (signals clipped to [0, 1])
        long_short — trade-by-trade equity (signals in [-1, 0, 1])

    Fees are taken into account during every single backtest run.
    """

    def __init__(self, exchange: str = 'binance', data_dir: str = './data'):
        self.bt = Backtester(exchange, data_dir)

    # ─────────────────────── Parameter grid ────────────────────────────

    @staticmethod
    def _build_grid(param_grid: Dict) -> Tuple[List[str], List[list], int]:
        """
        Build parameter ranges from a grid specification.

        param_grid values can be:
            (min, max, step)       — 3-element tuple, generates range
            (v1, v2, v3, v4, ...)  — tuple with != 3 elements, used as-is
            [v1, v2, ...]          — list, used as-is
            single_value           — wrapped in a list

        Returns (param_names, list_of_ranges, total_combo_count).
        """
        names = list(param_grid.keys())
        ranges = []
        for name in names:
            spec = param_grid[name]
            if isinstance(spec, (list, tuple)):
                if isinstance(spec, tuple) and len(spec) == 3:
                    lo, hi, step = spec
                    if isinstance(lo, float) or isinstance(hi, float) or isinstance(step, float):
                        vals = []
                        v = lo
                        while v <= hi + step * 1e-9:
                            vals.append(round(v, 10))
                            v += step
                    else:
                        vals = list(range(lo, hi + 1, step))
                    ranges.append(vals)
                else:
                    ranges.append(list(spec))
            else:
                ranges.append([spec])

        n_total = sum(len(r) for r in ranges)

        return names, ranges, n_total

    # ────────────────────── Core optimiser ─────────────────────────────

    def optimise(
        self,
        symbols:         List[str],
        strategy_func:   Callable,
        timeframe:       str,
        param_grid:      Dict,
        opt_start:       str   = _DEFAULT_START,
        opt_end:         str   = _DEFAULT_END,
        mode:            str   = 'long_short',
        initial_capital: float = _DEFAULT_CAPITAL,
        fee_per_trade:   float = _DEFAULT_FEE,
        stop_loss:       float = _DEFAULT_SL,
        leverage:        float = 1.0,
        metric:          str   = 'sharpe',
        n_rounds:        int   = 1,
        show_plots:      bool  = True,
    ) -> Dict:
        """
        Run a stepwise optimisation.

        Parameters
        ----------
        symbols        : list of trading pairs
        strategy_func  : function(df, **params) -> df with 'signal' column
        timeframe      : OHLCV timeframe
        param_grid     : {param_name: (min, max, step) or (v1,v2,...) or [values]}
        opt_start/end  : optimisation period
        mode           : 'long_only' or 'long_short'
        metric         : target metric to maximise
                         'sharpe'       — sqrt(N)*mean(R)/std(R)  (same as backtest.py)
                         'adj_sharpe'   — sharpe × sqrt(N)        (trade-count adjusted)
                         'sortino'      — sqrt(N)*mean(R)/downside_std(R)
                         'omega', 'ev', 'total_return', 'win_rate'
        n_rounds       : number of full sweeps through all parameters
        show_plots     : whether to display plots

        Returns
        -------
        dict with keys: best_per_symbol, global_best, history
        """
        assert mode in ('long_only', 'long_short', 'long_short_bar'), \
            "mode must be 'long_only', 'long_short', or 'long_short_bar'"

        param_names, ranges, n_evals_per_round = self._build_grid(param_grid)

        self._print_header(opt_start, opt_end, mode, len(symbols),
                           n_evals_per_round, n_rounds, param_names, metric)

        # ── Load data ─────────────────────────────────────────────────
        print('\n  Loading data...')
        data_cache = {}
        for symbol in symbols:
            print(f'  → {symbol}')
            df = self.bt.get_data(symbol, timeframe, opt_start, opt_end)
            data_cache[symbol] = df[['open', 'high', 'low', 'close', 'volume']].copy()
            del df
        gc.collect()

        # ── Default params (strategy function defaults, fallback to first range value) ──
        sig_defaults = {
            k: v.default
            for k, v in inspect.signature(strategy_func).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        current_best = {
            name: sig_defaults.get(name, rng[0])
            for name, rng in zip(param_names, ranges)
        }

        # Evaluate default
        default_score = self._evaluate_params(
            current_best, symbols, data_cache, strategy_func,
            mode, initial_capital, fee_per_trade, stop_loss, leverage, metric)
        print(f'\n  Default params:  {self._fmt_params(current_best)}')
        print(f'  Default avg {metric}: {default_score:.4f}')

        # ── Stepwise optimisation ─────────────────────────────────────
        history = []          # track progress across rounds
        best_per_symbol = {}
        n_workers = max(1, os.cpu_count() or 4)

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            for rnd in range(1, n_rounds + 1):
                print(f'\n{"─" * 70}')
                print(f'  ROUND {rnd}/{n_rounds}')
                print(f'{"─" * 70}')

                for p_idx, (p_name, p_range) in enumerate(zip(param_names, ranges)):
                    if len(p_range) <= 1:
                        continue  # nothing to optimise

                    print(f'\n  Sweeping  {p_name}  ({len(p_range)} values, {n_workers} workers) ...')
                    t0 = _time.time()

                    best_val   = current_best[p_name]
                    best_score = float('-inf')

                    # Build all trial param sets and submit to process pool
                    future_to_value = {}
                    for value in p_range:
                        trial_params = {**current_best, p_name: value}
                        fut = pool.submit(
                            _worker_evaluate,
                            strategy_func, data_cache, trial_params, symbols,
                            mode, initial_capital, fee_per_trade, stop_loss,
                            leverage, metric)
                        future_to_value[fut] = value

                    total = len(future_to_value)
                    done_count = 0
                    for fut in concurrent.futures.as_completed(future_to_value):
                        done_count += 1
                        value = future_to_value[fut]
                        try:
                            score = fut.result()
                        except Exception:
                            score = float('-inf')

                        if score > best_score:
                            best_score = score
                            best_val = value

                        # Progress bar
                        pct = done_count / total
                        elapsed = _time.time() - t0
                        eta = (elapsed / done_count) * (total - done_count) if done_count > 0 else 0
                        bar_len = 30
                        filled = int(bar_len * pct)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f'\r    {bar}  {pct:6.1%}  '
                              f'({done_count}/{total})  '
                              f'ETA {eta:.0f}s', end='', flush=True)

                    print()  # newline after progress bar

                    current_best[p_name] = best_val
                    print(f'  ✓ {p_name} = {best_val}  (avg {metric}: {best_score:.4f})')

                    history.append({
                        'round': rnd,
                        'param': p_name,
                        'value': best_val,
                        'avg_metric': best_score,
                        'params_snapshot': dict(current_best),
                    })

                    gc.collect()

        # ── Final evaluation per symbol with the best params ──────────
        print(f'\n{"─" * 70}')
        print(f'  Final evaluation with best params')
        print(f'{"─" * 70}')

        global_best = self._final_evaluation(
            current_best, symbols, data_cache, strategy_func,
            mode, initial_capital, fee_per_trade, stop_loss, leverage,
            metric, best_per_symbol)

        # ── Plots ────────────────────────────────────────────────────
        if show_plots:
            self._plot_results(
                best_per_symbol, global_best,
                data_cache, strategy_func, mode,
                initial_capital, fee_per_trade, stop_loss, leverage)

        return {
            'best_per_symbol': best_per_symbol,
            'global_best':     global_best,
            'history':         history,
        }

    # ─────────────────── Evaluation helpers ────────────────────────────

    @staticmethod
    def _enrich_metrics(metrics, trade_rets):
        """Add derived metrics to a metrics dict.

        sharpe      = backtester's value (sqrt(N)*mean/std), unchanged
        adj_sharpe  = sharpe × sqrt(N)  — rewards higher trade count
        sortino     = sqrt(N) * mean(R) / downside_std(R)
        """
        sharpe = metrics.get('sharpe', float('nan'))
        trades = metrics.get('trades', 0)
        if np.isfinite(sharpe) and trades > 0:
            metrics['adj_sharpe'] = sharpe * (trades ** 0.5)
        else:
            metrics['adj_sharpe'] = float('-inf')

        tr = np.asarray(trade_rets, dtype=float)
        if len(tr) >= 2:
            downside = tr[tr < 0]
            down_std = np.std(downside, ddof=0) if len(downside) > 0 else 0.0
            if down_std > 0:
                metrics['sortino'] = float(np.sqrt(len(tr)) * tr.mean() / down_std)
            else:
                metrics['sortino'] = float('inf') if tr.mean() > 0 else 0.0
        else:
            metrics['sortino'] = 0.0
        return metrics

    def _evaluate_params(self, params, symbols, data_cache, strategy_func,
                         mode, initial_capital, fee_per_trade, stop_loss, leverage, metric):
        """Run a single param set across all symbols, return avg metric."""
        values = []
        for symbol in symbols:
            df_base = data_cache.get(symbol)
            if df_base is None or df_base.empty:
                continue
            try:
                result_df = strategy_func(df_base.copy(), **params)
                result_df, trade_rets = self.bt._run_equity(
                    result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
                metrics = self._enrich_metrics(self.bt._compute_metrics(result_df, trade_rets), trade_rets)
                v = metrics.get(metric, float('nan'))
                if np.isfinite(v):
                    values.append(v)
                del result_df, trade_rets
            except Exception:
                pass
        return np.mean(values) if values else float('-inf')

    def _final_evaluation(self, best_params, symbols, data_cache, strategy_func,
                          mode, initial_capital, fee_per_trade, stop_loss, leverage,
                          metric, best_per_symbol):
        """Evaluate the final best params on every symbol and build global summary."""
        all_m = {k: [] for k in ['sharpe', 'adj_sharpe', 'sortino', 'omega', 'ev',
                                  'max_drawdown', 'total_return', 'win_rate']}

        for symbol in symbols:
            df_base = data_cache.get(symbol)
            if df_base is None or df_base.empty:
                print(f'\n  ✗ No data for {symbol}')
                continue
            try:
                result_df = strategy_func(df_base.copy(), **best_params)
                result_df, trade_rets = self.bt._run_equity(
                    result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
                metrics = self._enrich_metrics(self.bt._compute_metrics(result_df, trade_rets), trade_rets)
                metrics['params'] = dict(best_params)
                best_per_symbol[symbol] = metrics
                self._print_best(symbol, metrics, metric)
                for k in all_m:
                    v = metrics.get(k, float('nan'))
                    if np.isfinite(v):
                        all_m[k].append(v)
                del result_df, trade_rets
            except Exception as e:
                print(f'\n  ✗ {symbol}: {e}')
            gc.collect()

        metric_key = metric.replace('avg_', '')
        metric_vals = all_m.get(metric_key, [])
        avg_metric = np.mean(metric_vals) if metric_vals else float('nan')

        global_best = {
            'params':     dict(best_params),
            'avg_metric': avg_metric,
            'n_symbols':  len(metric_vals),
        }
        for k, v_list in all_m.items():
            global_best[f'avg_{k}'] = np.mean(v_list) if v_list else float('nan')

        self._print_global_best(global_best, metric, len(symbols))
        return global_best

    # ──────────────────────── Display ──────────────────────────────────

    @staticmethod
    def _fmt_params(params):
        return '  '.join(f'{k}={v}' for k, v in params.items())

    def _print_header(self, start, end, mode, n_sym, n_evals, n_rounds,
                       param_names, metric):
        W = 70
        tags = {'long_only': 'LONG-ONLY', 'long_short': 'LONG+SHORT', 'long_short_bar': 'LONG+SHORT BAR'}
        tag = tags.get(mode, mode.upper())
        print(f'\n┌{"─" * (W - 2)}┐')
        print(f'│  {"STEPWISE OPTIMISER  ·  " + tag:<{W - 4}}│')
        print(f'│  {f"{start}  →  {end}":<{W - 4}}│')
        estr = f'{n_sym} symbols  ·  {n_evals} evals/round  ·  {n_rounds} round(s)'
        print(f'│  {estr:<{W - 4}}│')
        pstr = f'Params: {", ".join(param_names)}'
        if len(pstr) > W - 6:
            pstr = pstr[:W - 9] + '...'
        print(f'│  {pstr:<{W - 4}}│')
        mstr = f'Target: {metric}'
        print(f'│  {mstr:<{W - 4}}│')
        print(f'└{"─" * (W - 2)}┘')

    def _print_best(self, symbol, best, metric):
        W = 70
        params_str   = '  '.join(f'{k}={v}' for k, v in best['params'].items())

        def _fs(key, signed=False):
            v = best.get(key, float('nan'))
            if not np.isfinite(v):
                return 'N/A'
            return f'{v:+.2f}' if signed else f'{v:.2f}'

        def _fp(key, signed=False):
            v = best.get(key, float('nan'))
            if not np.isfinite(v):
                return 'N/A'
            return f'{v:+.2%}' if signed else f'{v:.2%}'

        print(f'\n  ┌{"─" * (W - 4)}┐')
        print(f'  │  {"BEST  ·  " + symbol:<{W - 6}}│')
        while params_str:
            chunk = params_str[:W - 8]
            print(f'  │  {chunk:<{W - 6}}│')
            params_str = params_str[W - 8:]
        print(f'  ├{"─" * (W - 4)}┤')
        print(f'  │  {"Return":<13}{_fp("total_return", True):<19}{"Trades":<14}{str(best["trades"]):<18}│')
        print(f'  │  {"Max DD":<13}{_fp("max_drawdown"):<19}{"EV":<14}{_fp("ev", True):<18}│')
        print(f'  │  {"Sharpe":<13}{_fs("sharpe"):<19}{"Omega":<14}{_fs("omega"):<18}│')
        print(f'  │  {"Adj Sharpe":<13}{_fs("adj_sharpe"):<19}{"Sortino":<14}{_fs("sortino"):<18}│')
        print(f'  │  {"Win Rate":<13}{_fp("win_rate"):<19}{"Lose Streak":<14}{str(best["max_lose_streak"]):<18}│')
        print(f'  └{"─" * (W - 4)}┘')

    def _print_global_best(self, best, metric, n_symbols):
        W = 70
        params_str = '  '.join(f'{k}={v}' for k, v in best['params'].items())

        print(f'\n┌{"─" * (W - 2)}┐')
        sub = f'avg {metric} across {n_symbols} symbols'
        print(f'│  {"GLOBAL BEST  ·  " + sub:<{W - 4}}│')
        print(f'├{"─" * (W - 2)}┤')
        while params_str:
            chunk = params_str[:W - 6]
            print(f'│  {chunk:<{W - 4}}│')
            params_str = params_str[W - 6:]
        avg_val = best['avg_metric']
        line = f'Avg {metric}: {avg_val:.4f}'
        print(f'│  {line:<{W - 4}}│')

        # Print all average metrics
        for k in ['avg_sharpe', 'avg_adj_sharpe', 'avg_sortino', 'avg_omega', 'avg_ev',
                  'avg_max_drawdown', 'avg_total_return', 'avg_win_rate']:
            v = best.get(k, float('nan'))
            if np.isfinite(v):
                label = k.replace('avg_', '').replace('_', ' ').title()
                if 'drawdown' in k or 'ev' in k or 'return' in k or 'win' in k:
                    fmt = f'Avg {label}: {v:.2%}'
                else:
                    fmt = f'Avg {label}: {v:.4f}'
                print(f'│  {fmt:<{W - 4}}│')

        print(f'└{"─" * (W - 2)}┘')

    # ────────────────── Per-symbol optimisation ────────────────────────

    def optimise_per_symbol(
        self,
        symbols:         List[str],
        strategy_func:   Callable,
        timeframe:       str,
        param_grid:      Dict,
        opt_start:       str   = _DEFAULT_START,
        opt_end:         str   = _DEFAULT_END,
        mode:            str   = 'long_short',
        initial_capital: float = _DEFAULT_CAPITAL,
        fee_per_trade:   float = _DEFAULT_FEE,
        stop_loss:       float = _DEFAULT_SL,
        leverage:        float = 1.0,
        metric:          str   = 'sharpe',
        n_rounds:        int   = 1,
        show_plots:      bool  = True,
    ) -> Dict:
        """
        Optimise each symbol independently, then display all results together.

        Returns dict with per-symbol best params and metrics.
        """
        assert mode in ('long_only', 'long_short', 'long_short_bar')

        param_names, ranges, n_evals_per_round = self._build_grid(param_grid)

        # ── Load data once for all symbols ─────────────────────────
        print(f'\n┌{"─" * 68}┐')
        print(f'│  {"PER-SYMBOL STEPWISE OPTIMISER":<66}│')
        print(f'│  {f"{opt_start}  →  {opt_end}":<66}│')
        print(f'│  {f"{len(symbols)} symbols  ·  {n_evals_per_round} evals/round  ·  {n_rounds} round(s)":<66}│')
        print(f'│  {f"Target: {metric}":<66}│')
        print(f'└{"─" * 68}┘')

        print('\n  Loading data...')
        data_cache = {}
        for symbol in symbols:
            print(f'  → {symbol}')
            df = self.bt.get_data(symbol, timeframe, opt_start, opt_end)
            data_cache[symbol] = df[['open', 'high', 'low', 'close', 'volume']].copy()
            del df
        gc.collect()

        # ── Default params from strategy signature ─────────────────
        sig_defaults = {
            k: v.default
            for k, v in inspect.signature(strategy_func).parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        n_workers = max(1, os.cpu_count() or 4)
        per_symbol_results = {}

        # ── Optimise each symbol independently ─────────────────────
        for sym_idx, symbol in enumerate(symbols):
            df_base = data_cache.get(symbol)
            if df_base is None or df_base.empty:
                print(f'\n  ✗ No data for {symbol}, skipping')
                continue

            print(f'\n{"═" * 70}')
            print(f'  [{sym_idx + 1}/{len(symbols)}]  Optimising {symbol}')
            print(f'{"═" * 70}')

            current_best = {
                name: sig_defaults.get(name, rng[0])
                for name, rng in zip(param_names, ranges)
            }

            # Evaluate default for this symbol
            default_score = self._evaluate_params(
                current_best, [symbol], data_cache, strategy_func,
                mode, initial_capital, fee_per_trade, stop_loss, leverage, metric)
            print(f'  Default {metric}: {default_score:.4f}')

            # Stepwise sweep for this symbol only
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                for rnd in range(1, n_rounds + 1):
                    if n_rounds > 1:
                        print(f'\n  Round {rnd}/{n_rounds}')

                    for p_name, p_range in zip(param_names, ranges):
                        if len(p_range) <= 1:
                            continue

                        print(f'    Sweeping {p_name} ({len(p_range)} values) ...', end='', flush=True)
                        t0 = _time.time()

                        best_val = current_best[p_name]
                        best_score = float('-inf')

                        future_to_value = {}
                        for value in p_range:
                            trial_params = {**current_best, p_name: value}
                            fut = pool.submit(
                                _worker_evaluate,
                                strategy_func, data_cache, trial_params, [symbol],
                                mode, initial_capital, fee_per_trade, stop_loss,
                                leverage, metric)
                            future_to_value[fut] = value

                        for fut in concurrent.futures.as_completed(future_to_value):
                            value = future_to_value[fut]
                            try:
                                score = fut.result()
                            except Exception:
                                score = float('-inf')
                            if score > best_score:
                                best_score = score
                                best_val = value

                        elapsed = _time.time() - t0
                        current_best[p_name] = best_val
                        print(f'  {p_name}={best_val}  ({best_score:.4f})  [{elapsed:.1f}s]')

                    gc.collect()

            # Final eval for this symbol
            try:
                result_df = strategy_func(df_base.copy(), **current_best)
                result_df, trade_rets = self.bt._run_equity(
                    result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
                metrics = self._enrich_metrics(
                    self.bt._compute_metrics(result_df, trade_rets), trade_rets)
                metrics['params'] = dict(current_best)
                metrics['equity_df'] = result_df[['equity']].copy()
                per_symbol_results[symbol] = metrics
                del result_df, trade_rets
            except Exception as e:
                print(f'  ✗ Final eval failed: {e}')
            gc.collect()

        # ── Print all per-symbol results ───────────────────────────
        print(f'\n{"═" * 70}')
        print(f'  RESULTS — ALL SYMBOLS')
        print(f'{"═" * 70}')
        for symbol, metrics in per_symbol_results.items():
            self._print_best(symbol, metrics, metric)

        # ── Global averages ────────────────────────────────────────
        agg_keys = ['sharpe', 'adj_sharpe', 'sortino', 'omega', 'ev',
                     'max_drawdown', 'total_return', 'win_rate']
        all_m = {k: [] for k in agg_keys}
        for metrics in per_symbol_results.values():
            for k in agg_keys:
                v = metrics.get(k, float('nan'))
                if np.isfinite(v):
                    all_m[k].append(v)

        W = 70
        print(f'\n┌{"─" * (W - 2)}┐')
        print(f'│  {"AVERAGES ACROSS ALL SYMBOLS":<{W - 4}}│')
        print(f'├{"─" * (W - 2)}┤')
        for k, v_list in all_m.items():
            if v_list:
                avg = np.mean(v_list)
                label = k.replace('_', ' ').title()
                if 'drawdown' in k or 'ev' in k or 'return' in k or 'win' in k:
                    line = f'Avg {label}: {avg:.2%}'
                else:
                    line = f'Avg {label}: {avg:.4f}'
                print(f'│  {line:<{W - 4}}│')
        print(f'└{"─" * (W - 2)}┘')

        # ── Plot all equities ──────────────────────────────────────
        if show_plots:
            equities = {s: m['equity_df'] for s, m in per_symbol_results.items()
                        if 'equity_df' in m}
            self._plot_multi_equity(equities, 'Per-Symbol Optimised Equity Curves')

        # Clean up equity_df from returned results
        for m in per_symbol_results.values():
            m.pop('equity_df', None)

        return per_symbol_results

    # ──────────────────────── Plotting ─────────────────────────────────

    @staticmethod
    def _plot_multi_equity(equities: Dict, title: str):
        """All equity curves on one log-scale chart."""
        if not equities:
            return
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(equities), 10)))

        for (symbol, eq), color in zip(equities.items(), colors):
            short = symbol.split('/')[0]
            final = eq['equity'].iloc[-1]
            init  = eq['equity'].iloc[0]
            ret   = (final / init - 1) * 100
            ax.plot(eq.index, eq['equity'].values,
                    linewidth=2, alpha=0.8, color=color,
                    label=f'{short}  {ret:+.1f}%')

        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, which='both', alpha=0.2, linestyle='--')
        ax.grid(True, which='major', alpha=0.4)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        ax.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        plt.show()

    def _plot_results(self, best_per_symbol, global_best,
                      data_cache, strategy_func, mode,
                      initial_capital, fee_per_trade, stop_loss, leverage):
        """Equity curves: all symbols on one chart with the best params."""
        equities = {}
        params_str = '  '.join(
            f'{k}={v}' for k, v in global_best['params'].items())
        for symbol in data_cache:
            df = data_cache[symbol]
            try:
                result_df = strategy_func(df.copy(), **global_best['params'])
                result_df, _ = self.bt._run_equity(
                    result_df, mode, initial_capital, fee_per_trade, stop_loss, leverage)
                equities[symbol] = result_df[['equity']].copy()
                del result_df
                gc.collect()
            except Exception:
                pass
        self._plot_multi_equity(
            equities, f'Best Params ({params_str}) — Equity')
        del equities
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
#  Usage example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from strategies.hurst import hurst_strategy
    from strategies.lowvol_strategy import lowvol_strategy
    from strategies.genesis_v1 import genesis_v1

    opt = StrategyOptimiser()

    results = opt.optimise(
        symbols=['SUI/USDT', 'ETH/USDT', 'XRP/USDT', 'DOT/USDT', 'ZEC/USDT', 'LTC/USDT', 'ADA/USDT', 'UNI/USDT', 'LINK/USDT', 'AVAX/USDT'],
        strategy_func=lowvol_strategy,
        timeframe='1h',
        param_grid={
            # # MAD Trend
            # 'mad_len':          (15, 40, 5),
            # 'mad_mult':         [0.7, 0.85, 0.95, 1.1, 1.3],
            # # Median SD
            # 'med_dema_len':     (12, 36, 6),
            # 'med_rolling_len':  (20, 50, 5),
            # 'med_atr_mult':     [1.6, 2.0, 2.4, 2.8, 3.2],
            # 'med_atr_len':      (15, 40, 5),
            # 'med_sd_len':       (7, 21, 7),
            # # Zero-Lag SD
            # 'zl_dema_len':      (7, 21, 7),
            # 'zl_sd_len':        (15, 45, 5),
            # 'zl_upper_mult':    [1.010, 1.015, 1.018, 1.022, 1.030],
            # 'zl_lower_mult':    [0.970, 0.980, 0.988, 0.992, 0.995],
            # # EMA Base signal
            # 'ema_len':          (2, 8, 1),
            # 'ema_atr_len':      (15, 40, 5),
            # 'ema_mult':         [0.8, 1.0, 1.3, 1.6, 2.0],
            # # EMA Momentum signal
            # 'ema2_len':         (6, 20, 2),
            # 'ema2_atr_len':     (8, 25, 5),
            # 'ema2_mult':        [1.5, 2.0, 2.5, 3.0, 3.5],
            # # MAD signal
            # 'mad_sig_len':      (15, 40, 5),
            # 'mad_sig_mult':     [0.4, 0.55, 0.7, 0.85, 1.0],
            # # RSI signal
            # 'rsi_len':          (7, 20, 3),
            # 'rsi_sma_len':      (20, 45, 5),
            # # RMSD Trend
            # 'rmsd_len':         (15, 45, 5),
            # 'rmsd_mult':        [0.7, 0.9, 1.1, 1.3, 1.5],
            # # TPI thresholds
            # 'tpi_long':         [0.1, 0.2, 0.3, 0.4, 0.5],
            # 'tpi_short':        [-0.3, -0.2, -0.1, 0.0, 0.1],
            # ADF (rolling stationarity test)
            'adf_window':           (50, 500, 50),
            'adf_threshold':        (-3.0, 1, 0.25),
            # ADX (trend strength)
            'adx_period':           (10, 50, 5),
            'adx_threshold':        (10, 50, 5),
            'ema_small':            (10, 40, 5),
            'ema_big':              (20, 60, 5),
            'rsi_length':           (10, 40, 5),
            'rsi_long_threshold':   (60, 80, 5),
            'rsi_short_threshold':  (20, 40, 5),
            'er_period':            (200, 800, 50),
            'er_threshold':         (3, 15, 1),
            'rs_length':            (10, 50, 5),
            'rs_lookback':          (100, 300, 50),
            # 'garch_threshold_long':      (30, 100, 10),
            # 'garch_threshold_short':      (30, 100, 10),
            # 'garch_threshold':      (30, 100, 10),
            'klass_threshold':      (30, 100, 10),
        },
        opt_start='2023-01-01',
        opt_end='2025-01-01',
        mode='long_short',
        fee_per_trade=0.0005,
        stop_loss=0.1,
        metric='adj_sharpe',
        n_rounds=1
    )
