import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from xgboost import XGBClassifier, XGBRegressor

from strategies.hurst import hurst_strategy
from utils.backtest import Backtester

# ============================================================
# CONFIGURATION
# ============================================================
opt_start_date = '2023-01-01'   # Start of optimization / training
opt_end_date = '2023-01-01'     # End of optimization / training
bt_start_date = '2023-01-01'    # Start of out-of-sample backtest
bt_end_date = datetime.now().date()

start_date = opt_start_date
end_date = bt_end_date
initial_capital = 10000
exposure_per_trade = 1.0   # 1.0 = 100% of capital per trade
fee_per_trade = 0.0005     # 0.04% per trade (entry + exit = 0.08% round trip)
stop_loss_pct = 0.05       # 5% stop loss distance

# Risk management
max_concurrent_trades = 10  # Max number of coins traded at once
daily_loss_limit = 0.03    # 5% max daily loss (set to 0 to disable)
weekly_loss_limit = 0.1   # 10% max weekly loss (set to 0 to disable)
monthly_loss_limit = 0.20  # 20% max monthly loss (set to 0 to disable)


class Backtester:
    def __init__(self, exchange='binance', data_dir='./data'):
        self.exchange = getattr(ccxt, exchange)()
        # Initialize with 'swap' defaultType to fetch perpetual futures data
        # self.exchange = getattr(ccxt, exchange)({
        #     'options': {
        #         'defaultType': 'swap',
        #     }
        # })
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.feather"
        # return self.data_dir / f"{symbol.replace('/', '_')}_futures_{timeframe}.feather"
    
    def get_data(self, symbol: str, timeframe: str, 
                 start_date = start_date, end_date = end_date,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Smart data fetcher with local cache"""
        cache_file = self._cache_path(symbol, timeframe)
        
        # Load existing cache
        if cache_file.exists() and not force_refresh:
            df = pd.read_feather(cache_file).set_index('timestamp')
            
            # Check if we have enough data starting from start_date
            if not df.empty and df.index[0] <= pd.Timestamp(start_date):
                # Check if the cache is reasonably fresh (within 6 hours of end_date)
                has_end = not end_date or df.index[-1] >= pd.Timestamp(end_date) - pd.Timedelta(hours=6)
                
                if has_end:
                    df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    return df
                else:
                    # Cache exists but is stale — only download the missing tail
                    return self._download_with_cache(symbol, timeframe, cached_df=df)
        
        # No cache at all, or cache doesn't cover start_date — full download
        return self._download_with_cache(symbol, timeframe)
    
    def _download_with_cache(self, symbol: str, timeframe: str, cached_df: pd.DataFrame = None) -> pd.DataFrame:
        """Download and merge with existing cache"""
        cache_file = self._cache_path(symbol, timeframe)
        
        # Find exactly where to start downloading from
        since_date = pd.Timestamp(start_date)
        if cached_df is None and cache_file.exists():
            cached_df = pd.read_feather(cache_file).set_index('timestamp')
        
        if cached_df is not None and not cached_df.empty and cached_df.index[0] <= since_date:
            since_date = cached_df.index[-1]
            
        # Download fresh data
        new_df = self._fetch_from_exchange(symbol, timeframe, since_date)
        if new_df.empty:
            if cached_df is not None and not cached_df.empty:
                result_df = cached_df[cached_df.index >= pd.Timestamp(start_date)]
                if end_date:
                    result_df = result_df[result_df.index <= pd.Timestamp(end_date)]
                return result_df
            return new_df
        
        # Merge with existing cache
        if cached_df is not None and not cached_df.empty:
            # Remove overlap
            cached_df = cached_df[cached_df.index < new_df.index[0]]
            
            # Combine
            combined_df = pd.concat([cached_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = new_df
        
        # Save updated cache
        combined_df.reset_index().to_feather(cache_file)
        
        # Return requested range
        result_df = combined_df[combined_df.index >= pd.Timestamp(start_date)]
        if end_date:
            result_df = result_df[result_df.index <= pd.Timestamp(end_date)]
        
        return result_df
    
    def _fetch_from_exchange(self, symbol: str, timeframe: str, since_date=None) -> pd.DataFrame:
        """Raw data download from exchange"""
        if since_date:
            since = int(since_date.timestamp() * 1000)
        else:
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
        all_ohlcv = []
        current_since = since
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break
            
            if not ohlcv or len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Move forward
            current_since = ohlcv[-1][0] + 1
            
            # Break if we have enough data
            if end_date:
                last_time = pd.to_datetime(ohlcv[-1][0], unit='ms')
                if last_time >= pd.Timestamp(end_date):
                    break
            
            if len(ohlcv) < 1000:
                break
            
            # Rate limiting
            time.sleep(self.exchange.rateLimit / 1000)
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df
    
    def run_strategy(self, df: pd.DataFrame, strategy_func: Callable) -> pd.DataFrame:
        """Run strategy on cached data"""
        df = strategy_func(df.copy())
        df = self._calculate_positions(df)
        df = self._calculate_equity(df, initial_capital=10000)
        return df
    
    def _calculate_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert signals directly to target positions (supports long, short, cash)"""
        df['position'] = df['signal']
        return df
    
    def _calculate_equity(self, df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """Calculate equity trade-by-trade"""
        close = df['close'].values
        positions = df['position'].values
        
        equity = np.full(len(df), initial_capital)
        strategy_returns = np.zeros(len(df))
        curr_equity = initial_capital
        
        entry_price = 0.0
        
        for i in range(1, len(df)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            if prev_pos != curr_pos:
                if prev_pos != 0:
                    exit_price = close[i]
                    if prev_pos == 1:
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price
                        
                    curr_equity *= (1 + ret)
                    strategy_returns[i] = ret
                    
                if curr_pos != 0:
                    entry_price = close[i]
                    
            equity[i] = curr_equity
            
        df['strategy_returns'] = strategy_returns
        df['equity'] = equity
        df['drawdown'] = df['equity'] / df['equity'].cummax() - 1
        return df
    
    def backtest_portfolio(self, symbols: List[str], strategy_func: Callable,
                           timeframe: str, start_date=start_date, end_date=end_date,
                           exposure=exposure_per_trade, initial_capital=initial_capital,
                           stop_loss=stop_loss_pct, eval_start_date=None,
                           _precomputed_data=None,
                           max_trades=max_concurrent_trades,
                           daily_limit=daily_loss_limit,
                           weekly_limit=weekly_loss_limit,
                           monthly_limit=monthly_loss_limit,
                           coin_params: Optional[Dict[str, Dict]] = None,
                           show_plot: bool = True) -> Dict:
        """
        Portfolio backtest: runs strategy on all coins simultaneously.
        Each coin can open a position independently. Multiple positions
        can be active at the same time. Each position uses `exposure` 
        fraction of current equity at entry time.
        """
        # 1. Pre-download all data
        print("=" * 50)
        print("Pre-loading data for all symbols...")
        raw_data = {}
        for symbol in symbols:
            print(f"  Fetching {symbol}...")
            df = self.get_data(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                raw_data[symbol] = df
            else:
                print(f"  ⚠ No data for {symbol}")
        
        # 2. Run strategy on each coin independently (unless precomputed)
        if _precomputed_data:
            strategy_data = _precomputed_data
        else:
            print("\n" + "=" * 50)
            print("Running strategy on each coin...")
            strategy_data = {}
            _coin_params = coin_params or {}
            for symbol, df in raw_data.items():
                print(f"  Processing {symbol}...")
                params = _coin_params.get(symbol, {})
                result = strategy_func(df.copy(), **params)
                result['position'] = result['signal']
                strategy_data[symbol] = result
        
        # 3. Build a unified time index from all coins
        all_indices = set()
        for df in strategy_data.values():
            all_indices.update(df.index)
        unified_index = sorted(all_indices)
        
        if eval_start_date is not None:
            eval_start_ts = pd.Timestamp(eval_start_date)
            unified_index = [idx for idx in unified_index if idx >= eval_start_ts]
        
        # 4. Reindex each coin's data to the unified index
        aligned = {}
        for symbol, df in strategy_data.items():
            aligned[symbol] = df.reindex(unified_index, method='ffill')
        
        # 5. Portfolio simulation — trade by trade across all coins
        n_bars = len(unified_index)
        portfolio_equity = np.full(n_bars, float(initial_capital))
        curr_equity = float(initial_capital)
        
        # Track active positions: {symbol: {'direction': 1/-1, 'entry_price': float, 'capital_allocated': float}}
        active_positions = {}
        all_trade_returns = []  # flat list of all individual trade returns
        long_returns  = []      # returns from long trades only
        short_returns = []      # returns from short trades only
        per_coin_trades = {s: [] for s in strategy_data}  # per-coin trade returns
        
        # Loss limit tracking: equity at the start of each period
        period_start_equity = {'daily': float(initial_capital),
                               'weekly': float(initial_capital),
                               'monthly': float(initial_capital)}
        current_day = unified_index[0].date()
        current_week = unified_index[0].isocalendar()[:2]   # (year, week)
        current_month = (unified_index[0].year, unified_index[0].month)

        for i in range(1, n_bars):
            ts = unified_index[i]

            # Reset period trackers on new day/week/month
            ts_date = ts.date()
            ts_week = ts.isocalendar()[:2]
            ts_month = (ts.year, ts.month)
            if ts_date != current_day:
                current_day = ts_date
                period_start_equity['daily'] = curr_equity
            if ts_week != current_week:
                current_week = ts_week
                period_start_equity['weekly'] = curr_equity
            if ts_month != current_month:
                current_month = ts_month
                period_start_equity['monthly'] = curr_equity

            # Check if any loss limit is breached (block new entries)
            loss_limited = False
            if daily_limit > 0 and curr_equity < period_start_equity['daily'] * (1 - daily_limit):
                loss_limited = True
            if weekly_limit > 0 and curr_equity < period_start_equity['weekly'] * (1 - weekly_limit):
                loss_limited = True
            if monthly_limit > 0 and curr_equity < period_start_equity['monthly'] * (1 - monthly_limit):
                loss_limited = True

            for symbol in strategy_data:
                df = aligned[symbol]
                prev_pos = df['position'].iloc[i-1] if not pd.isna(df['position'].iloc[i-1]) else 0
                curr_pos = df['position'].iloc[i] if not pd.isna(df['position'].iloc[i]) else 0
                close_price = df['close'].iloc[i]
                high_price = df['high'].iloc[i]
                low_price = df['low'].iloc[i]
                
                if pd.isna(close_price):
                    continue
                
                # Check for stop loss hit if we have an active position
                stop_hit = False
                if symbol in active_positions and stop_loss > 0:
                    pos = active_positions[symbol]
                    if pos['direction'] == 1:
                        sl_price = pos['entry_price'] * (1 - stop_loss)
                        if low_price <= sl_price:
                            stop_hit = True
                            exit_price = min(close_price, sl_price) # fill at SL or worse if gap
                    else:
                        sl_price = pos['entry_price'] * (1 + stop_loss)
                        if high_price >= sl_price:
                            stop_hit = True
                            exit_price = max(close_price, sl_price)

                # Process exits (either from strategy signal change or stop loss hit)
                if stop_hit or prev_pos != curr_pos:
                    if symbol in active_positions:
                        pos = active_positions[symbol]
                        
                        # Use SL price if stopped out, else use close
                        exit_price_actual = exit_price if stop_hit else close_price
                        
                        if pos['direction'] == 1:
                            ret = (exit_price_actual - pos['entry_price']) / pos['entry_price']
                        else:
                            ret = (pos['entry_price'] - exit_price_actual) / pos['entry_price']
                        
                        # Deduct fees (entry + exit)
                        ret -= 2 * fee_per_trade
                        
                        pnl = pos['capital_allocated'] * ret
                        curr_equity += pnl
                        all_trade_returns.append(ret)
                        per_coin_trades[symbol].append(ret)
                        if pos['direction'] == 1:
                            long_returns.append(ret)
                        else:
                            short_returns.append(ret)
                        del active_positions[symbol]
                        
                        # If stopped out, we override curr_pos to 0 so we don't open a new trade immediately
                        # unless the strategy happens to trigger a new entry on this exact same bar
                        if stop_hit:
                            curr_pos = 0 
                    
                    # Open new position if applicable (respect limits)
                    if curr_pos != 0 and prev_pos != curr_pos \
                            and not loss_limited \
                            and len(active_positions) < max_trades:
                        capital_for_trade = curr_equity * exposure
                        active_positions[symbol] = {
                            'direction': int(curr_pos),
                            'entry_price': close_price,
                            'capital_allocated': capital_for_trade
                        }
            
            # Trade-by-trade: equity only changes on trade close
            portfolio_equity[i] = curr_equity
        
        # 6. Build portfolio DataFrame
        portfolio_df = pd.DataFrame({
            'equity': portfolio_equity
        }, index=unified_index)
        portfolio_df['drawdown'] = portfolio_df['equity'] / portfolio_df['equity'].cummax() - 1
        
        # 7. Calculate metrics
        total_return = portfolio_df['equity'].iloc[-1] / initial_capital - 1
        max_dd = portfolio_df['drawdown'].min()
        days = (unified_index[-1] - unified_index[0]).days
        cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        trade_returns_arr = np.array(all_trade_returns) if all_trade_returns else np.array([])
        total_trades = len(trade_returns_arr)
        
        if total_trades > 0:
            ev = float(np.mean(trade_returns_arr))
            win_rate = float(np.mean(trade_returns_arr > 0))
            sharpe = float(np.sqrt(total_trades) * np.mean(trade_returns_arr) / np.std(trade_returns_arr)) if np.std(trade_returns_arr) > 0 else 0
            
            pos_sum = trade_returns_arr[trade_returns_arr > 0].sum()
            neg_sum = trade_returns_arr[trade_returns_arr < 0].sum()
            omega = float(pos_sum / abs(neg_sum)) if neg_sum != 0 else np.inf
            
            lose_streak = 0
            max_lose_streak = 0
            for r in trade_returns_arr:
                if r < 0:
                    lose_streak += 1
                    max_lose_streak = max(max_lose_streak, lose_streak)
                else:
                    lose_streak = 0
        else:
            ev = win_rate = sharpe = omega = 0.0
            max_lose_streak = 0
        
        # 8. Print results
        print("\n" + "=" * 50)
        print("PER-COIN BREAKDOWN")
        print("=" * 50)
        for symbol in symbols:
            trades = per_coin_trades.get(symbol, [])
            if trades:
                coin_ev = np.mean(trades)
                coin_wr = np.mean(np.array(trades) > 0)
                print(f"  {symbol:12s} | Trades: {len(trades):4d} | EV: {coin_ev:+.2%} | WR: {coin_wr:.2%}")
            else:
                print(f"  {symbol:12s} | Trades:    0")
        
        print("\n" + "=" * 50)
        print("PORTFOLIO RESULTS")
        print("=" * 50)
        print(f"   Return: {total_return:+.2%} | Max DD: {max_dd:.2%} | CAGR: {cagr:+.2%}")
        ev_long  = float(np.mean(long_returns))  if long_returns  else 0.0
        ev_short = float(np.mean(short_returns)) if short_returns else 0.0
        print(f"   Trades: {total_trades} | EV: {ev:+.2%} | Win Rate: {win_rate:.2%}")
        print(f"   EV Long: {ev_long:+.2%} ({len(long_returns)} trades) | EV Short: {ev_short:+.2%} ({len(short_returns)} trades)")
        print(f"   Sharpe: {sharpe:.2f} | Omega: {omega:.2f} | Lose Streak: {max_lose_streak}")
        print(f"   Exposure per trade: {exposure:.0%} | Fee: {fee_per_trade:.4%} per side | Stop Loss: {stop_loss:.2%}")
        print(f"   Max concurrent trades: {max_trades} | Loss limits: D={daily_limit:.0%} W={weekly_limit:.0%} M={monthly_limit:.0%}")
        print("=" * 50)
        
        # 9. Plot
        if show_plot and n_bars > 1:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(portfolio_df.index, portfolio_df['equity'].values,
                    linewidth=2, color='#2962FF', alpha=0.8)
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(True, which='both', alpha=0.2, linestyle='--')
            ax.grid(True, which='major', alpha=0.4)
            ax.set_title(f'Portfolio Equity ({len(symbols)} coins, {exposure:.0%} exposure)',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity (log $)', fontsize=12)
            
            final_eq = portfolio_df['equity'].iloc[-1]
            ax.text(0.02, 0.98,
                f'Final: ${final_eq:,.0f}\nReturn: {total_return:+.1%}\nMax DD: {max_dd:.1%}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        
        return {
            'portfolio_df': portfolio_df,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'cagr': cagr,
            'sharpe': sharpe,
            'omega': omega,
            'ev': ev,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'max_lose_streak': max_lose_streak,
            'per_coin_trades': per_coin_trades
        }

    def backtest_portfolio_rolling(self, symbols: List[str], strategy_func: Callable,
                                   timeframe: str, window_bars: int, step_bars: int,
                                   start_date=start_date, end_date=end_date,
                                   exposure=exposure_per_trade, initial_capital=initial_capital,
                                   stop_loss=stop_loss_pct,
                                   cross_coin: bool = False,
                                   max_coins: int = 5,
                                   max_trades=max_concurrent_trades,
                                   daily_limit=daily_loss_limit,
                                   weekly_limit=weekly_loss_limit,
                                   monthly_limit=monthly_loss_limit,
                                   coin_params: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        Rolling Walk-Forward Portfolio Backtest.
        Trains on historical chunks and stitches OOS predictions for all symbols.

        cross_coin : bool
            When True, strategy_func receives a dict of ALL coins' pre-processed
            DataFrames and trains a single pooled model per step (see
            ml_hurst_strategy_cross).  When False (default), each coin is
            processed independently as before.
        """
        # 1. Pre-download data
        print("=" * 50)
        print("Rolling Walk-Forward Portfolio Backtest")
        print(f"  Window: {window_bars} bars | Step: {step_bars} bars")
        print(f"  Mode: {'cross-coin' if cross_coin else 'per-coin'}")
        print("Pre-loading data...")
        raw_data = {}
        for symbol in symbols:
            print(f"  Fetching {symbol}...")
            df = self.get_data(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                raw_data[symbol] = df

        if not raw_data:
            print("No data found for any symbols.")
            return {}

        if cross_coin:
            strategy_data = self._rolling_cross_coin(
                raw_data, strategy_func, window_bars, step_bars, max_coins)
        else:
            strategy_data = self._rolling_per_coin(
                raw_data, strategy_func, window_bars, step_bars, coin_params)

        if not strategy_data:
            print("No strategy data produced.")
            return {}

        first_symbol = list(strategy_data.keys())[0]
        eval_start_dt = strategy_data[first_symbol].index[window_bars]

        return self.backtest_portfolio(
            symbols=list(strategy_data.keys()),
            strategy_func=None,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            exposure=exposure,
            initial_capital=initial_capital,
            stop_loss=stop_loss,
            eval_start_date=eval_start_dt,
            _precomputed_data=strategy_data,
            max_trades=max_trades,
            daily_limit=daily_limit,
            weekly_limit=weekly_limit,
            monthly_limit=monthly_limit
        )

    # ── rolling helpers ──────────────────────────────────────────────

    def _rolling_per_coin(self, raw_data, strategy_func, window_bars, step_bars, coin_params=None):
        """Original mode: each coin trains its own model independently."""
        print("\n" + "=" * 50)
        print("Generating rolling signals Symbol-by-Symbol...")
        strategy_data = {}

        _coin_params = coin_params or {}
        for symbol, df in raw_data.items():
            print(f"  Processing {symbol}...")
            total_bars = len(df)
            combined_signals = np.zeros(total_bars)
            n_steps = 0
            params = _coin_params.get(symbol, {})

            if total_bars <= window_bars:
                print(f"    ⚠ Not enough data for rolling window ({total_bars} bars)")
                continue

            for i in range(window_bars, total_bars, step_bars):
                end_i       = min(i + step_bars, total_bars)
                train_start = df.index[i - window_bars]
                train_end   = df.index[i - 1]
                pred_start  = df.index[i]
                pred_end    = df.index[end_i - 1]

                n_steps += 1
                print(f"  Step {n_steps}: Symbols {len(raw_data)} | "
                      f"Range [{str(pred_start)[:10]} → {str(pred_end)[:10]}]")

                res_df = strategy_func(df.copy(),
                                       opt_start=train_start, opt_end=train_end,
                                       bt_start=pred_start,   bt_end=pred_end,
                                       **params)
                chunk_mask = (df.index >= pred_start) & (df.index <= pred_end)
                combined_signals[chunk_mask] = res_df.loc[chunk_mask, 'signal'].values

            symbol_res = df.copy()
            symbol_res['signal']   = combined_signals
            symbol_res['position'] = combined_signals
            strategy_data[symbol]  = symbol_res

        return strategy_data

    def _rolling_cross_coin(self, raw_data, strategy_func, window_bars, step_bars, max_coins=5):
        """Cross-coin mode: ONE model trained on pooled data from all coins."""
        print("\n" + "=" * 50)
        print("Pre-computing hurst features for all coins...")
        base_data = {}
        for symbol, df in raw_data.items():
            print(f"  Features: {symbol}...")
            base_data[symbol] = _prepare_hurst_features(df)

        # drop coins too short for the window
        base_data = {s: df for s, df in base_data.items()
                     if len(df) > window_bars}
        if not base_data:
            print("  ⚠ No coin has enough data for the rolling window")
            return {}

        # use shortest coin for index boundaries
        ref_symbol = min(base_data, key=lambda s: len(base_data[s]))
        ref_df     = base_data[ref_symbol]
        min_bars   = len(ref_df)

        combined = {s: np.zeros(len(df)) for s, df in base_data.items()}
        n_steps  = 0

        print("\n" + "=" * 50)
        print("Generating rolling signals Cross-Coin...")

        for i in range(window_bars, min_bars, step_bars):
            end_i       = min(i + step_bars, min_bars)
            train_start = ref_df.index[i - window_bars]
            train_end   = ref_df.index[i - 1]
            pred_start  = ref_df.index[i]
            pred_end    = ref_df.index[end_i - 1]

            n_steps += 1
            print(f"  Step {n_steps}: {len(base_data)} coins | "
                  f"[{str(pred_start)[:10]} → {str(pred_end)[:10]}]")

            step_results = strategy_func(
                {s: df.copy() for s, df in base_data.items()},
                max_coins=max_coins,
                opt_start=train_start, opt_end=train_end,
                bt_start=pred_start,   bt_end=pred_end,
            )

            for symbol in base_data:
                if symbol not in step_results:
                    continue
                coin_idx   = base_data[symbol].index
                chunk_mask = (coin_idx >= pred_start) & (coin_idx <= pred_end)
                res_df     = step_results[symbol]
                res_mask   = (res_df.index >= pred_start) & (res_df.index <= pred_end)
                combined[symbol][chunk_mask] = res_df.loc[res_mask, 'signal'].values

        strategy_data = {}
        for symbol, df in base_data.items():
            res = df.copy()
            res['signal']   = combined[symbol]
            res['position'] = combined[symbol]
            strategy_data[symbol] = res

        return strategy_data

    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return np.sqrt(len(returns)) * returns.mean() / returns.std()
    
    def _calculate_cagr(self, df: pd.DataFrame, total_return: float) -> float:
        days = (df.index[-1] - df.index[0]).days
        if days == 0:
            return 0
        return (1 + total_return) ** (365 / days) - 1
    
    def _get_trade_metrics(self, trade_returns: pd.Series) -> Dict:
        if len(trade_returns) == 0:
            return {'ev': 0.0, 'win_rate': 0.0, 'max_lose_streak': 0}
            
        trade_returns = trade_returns.values
        
        lose_streak = 0
        max_lose_streak = 0
        for r in trade_returns:
            if r < 0:
                lose_streak += 1
                max_lose_streak = max(max_lose_streak, lose_streak)
            else:
                lose_streak = 0
                
        return {
            'ev': float(np.mean(trade_returns)),
            'win_rate': float(np.mean(trade_returns > 0)),
            'max_lose_streak': max_lose_streak
        }

    def _calculate_omega(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        excess_returns = returns - risk_free_rate
        positive_sum = excess_returns[excess_returns > 0].sum()
        negative_sum = excess_returns[excess_returns < 0].sum()
        if negative_sum == 0:
            return np.nan if positive_sum == 0 else np.inf
        return float(positive_sum / abs(negative_sum))


if __name__ == "__main__":
    bt = Backtester()
    
    # Toggle: 'rolling' or 'static'
    mode = 'static'
    
    symbols = ['SUI/USDT', 'ETH/USDT', 'XRP/USDT', 'DOT/USDT', 'ZEC/USDT', 'LTC/USDT', 'ADA/USDT', 'UNI/USDT', 'LINK/USDT', 'AVAX/USDT']
    

    # 2 thresholds for garch

    if mode == 'rolling':
        print(f"--- Rolling Cross-Coin Meta-Labeling Strategy ---")
        print(f"Data range: {opt_start_date} to {bt_end_date}\n")

        # 5m timeframe config
        WINDOW = 50000   # ~173 days training
        STEP   = 10000   # ~34 days step
        bt = Backtester()
        results = bt.backtest_portfolio_rolling(
            symbols=symbols,
            timeframe='5m',
            strategy_func=hurst_strategy,
            window_bars=WINDOW,
            step_bars=STEP,
            start_date=opt_start_date,
            end_date=bt_end_date,
            exposure=exposure_per_trade,
            initial_capital=initial_capital,
            stop_loss=stop_loss_pct,
            
        
        )
    else:
        print(f"--- Meta-Labeling Portfolio Strategy ---")
        print(f"Optimization Period: {opt_start_date} to {opt_end_date}")
        print(f"Backtest Period:     {bt_start_date} to {bt_end_date}\n")

        results = bt.backtest_portfolio(
            symbols=symbols,
            timeframe='5m',
            strategy_func=hurst_strategy,
            exposure=exposure_per_trade,
            stop_loss=stop_loss_pct,
            eval_start_date=bt_start_date,
            coin_params=coin_params,
        )


