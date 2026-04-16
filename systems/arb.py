import numpy as np
import pandas as pd
import ccxt
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# CONFIGURATION
# ============================================================
start_date = '2020-01-01'
end_date = datetime.now().date()
initial_capital = 10000
exposure_per_trade = 1.0
fee_per_trade = 0.0005
stop_loss_pct = 0.05

# Stat-arb specific
TIMEFRAME = '5m'
MOVE_THRESHOLD = 0.03               # 3% median move to trigger scan
MOVE_LOOKBACK_BARS = 48             # 48 x 5m = 4h
EMA_FAST = 12                       # trend filter
EMA_SLOW = 21
MAX_CONCURRENT_TRADES = 3           # max N trades open at once
SWING_LOOKBACK = 24                 # bars to find recent swing high/low for SL
LAG_ZSCORE_THRESHOLD = 1.5          # min Z-score of lag to qualify for entry


class Backtester:
    def __init__(self, exchange='binance', data_dir='./data'):
        self.exchange = getattr(ccxt, exchange)()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.feather"

    def get_data(self, symbol: str, timeframe: str,
                 start_date=start_date, end_date=end_date,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Smart data fetcher with local cache"""
        cache_file = self._cache_path(symbol, timeframe)

        if cache_file.exists() and not force_refresh:
            df = pd.read_feather(cache_file).set_index('timestamp')

            if not df.empty and df.index[0] <= pd.Timestamp(start_date):
                has_end = not end_date or df.index[-1] >= pd.Timestamp(end_date) - pd.Timedelta(hours=6)

                if has_end:
                    df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    return df
                else:
                    return self._download_with_cache(symbol, timeframe, cached_df=df)

        return self._download_with_cache(symbol, timeframe)

    def _download_with_cache(self, symbol: str, timeframe: str, cached_df: pd.DataFrame = None) -> pd.DataFrame:
        """Download and merge with existing cache"""
        cache_file = self._cache_path(symbol, timeframe)

        since_date = pd.Timestamp(start_date)
        if cached_df is None and cache_file.exists():
            cached_df = pd.read_feather(cache_file).set_index('timestamp')

        if cached_df is not None and not cached_df.empty and cached_df.index[0] <= since_date:
            since_date = cached_df.index[-1]

        new_df = self._fetch_from_exchange(symbol, timeframe, since_date)
        if new_df.empty:
            if cached_df is not None and not cached_df.empty:
                result_df = cached_df[cached_df.index >= pd.Timestamp(start_date)]
                if end_date:
                    result_df = result_df[result_df.index <= pd.Timestamp(end_date)]
                return result_df
            return new_df

        if cached_df is not None and not cached_df.empty:
            cached_df = cached_df[cached_df.index < new_df.index[0]]
            combined_df = pd.concat([cached_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = new_df

        combined_df.reset_index().to_feather(cache_file)

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
            current_since = ohlcv[-1][0] + 1

            if end_date:
                last_time = pd.to_datetime(ohlcv[-1][0], unit='ms')
                if last_time >= pd.Timestamp(end_date):
                    break

            if len(ohlcv) < 1000:
                break

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

    # ============================================================
    # STAT-ARB PORTFOLIO BACKTEST
    # ============================================================
    def backtest_stat_arb(self, symbols: List[str],
                          timeframe: str = TIMEFRAME,
                          start_date=start_date, end_date=end_date,
                          move_threshold: float = MOVE_THRESHOLD,
                          move_lookback: int = MOVE_LOOKBACK_BARS,
                          ema_fast: int = EMA_FAST,
                          ema_slow: int = EMA_SLOW,
                          max_trades: int = MAX_CONCURRENT_TRADES,
                          swing_lookback: int = SWING_LOOKBACK,
                          lag_zscore: float = LAG_ZSCORE_THRESHOLD,
                          exposure: float = exposure_per_trade,
                          initial_capital: float = initial_capital,
                          fee_per_trade: float = fee_per_trade) -> Dict:
        """
        Statistical Arbitrage Portfolio Backtest.

        Logic:
        1. Compute the median 4h move across all coins as the "market move"
        2. When |median move| > move_threshold, find the coin lagging the most vs the median
        3. Enter long/short accordingly with:
           - TP at the "fair value" (where the coin should be if it matched the median move)
           - SL at recent swing low/high
        4. EMA cross as trend filter
        5. Max N concurrent trades
        """
        # 1. Pre-download all data
        print("=" * 60)
        print("STATISTICAL ARBITRAGE BACKTEST")
        print("=" * 60)
        print(f"  Coins:      {', '.join(symbols)}")
        print(f"  Timeframe:  {timeframe}")
        print(f"  Move threshold: {move_threshold:.1%} (median of all coins) in {move_lookback} bars")
        print(f"  Max concurrent trades: {max_trades}")
        print(f"  EMA filter: {ema_fast}/{ema_slow}")
        print(f"  Lag Z-score threshold: {lag_zscore:.1f}")
        print("-" * 60)

        print("\nPre-loading data...")

        # Fetch all coins
        raw_data = {}
        for symbol in symbols:
            print(f"  Fetching {symbol}...")
            df = self.get_data(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                raw_data[symbol] = df
            else:
                print(f"  WARNING: No data for {symbol}")

        if not raw_data:
            print("No coin data available.")
            return {}

        # 2. Build unified index
        all_indices = set()
        for df in raw_data.values():
            all_indices.update(df.index)
        unified_index = sorted(all_indices)

        # 3. Reindex everything to unified index
        aligned = {}
        for symbol, df in raw_data.items():
            aligned[symbol] = df[['open', 'high', 'low', 'close', 'volume']].reindex(unified_index, method='ffill')

        # 4. Pre-compute indicators for all coins
        print("\nComputing indicators...")

        # Per-coin 4h moves
        all_moves_4h = {}
        coin_indicators = {}
        for symbol in raw_data:
            df = aligned[symbol]
            close = df['close']

            # EMA trend filter
            ema_f = close.ewm(span=ema_fast, adjust=False).mean()
            ema_s = close.ewm(span=ema_slow, adjust=False).mean()
            ema_bullish = ema_f > ema_s

            # Coin's 4h move
            coin_move_4h = close.pct_change(periods=move_lookback)
            all_moves_4h[symbol] = coin_move_4h

            # Swing highs/lows for stop loss
            swing_high = close.rolling(swing_lookback).max()
            swing_low = close.rolling(swing_lookback).min()

            coin_indicators[symbol] = {
                'close': close,
                'high': df['high'],
                'low': df['low'],
                'ema_bullish': ema_bullish,
                'move_4h': coin_move_4h,
                'swing_high': swing_high,
                'swing_low': swing_low,
            }

        # Median 4h move across all coins at each bar
        moves_df = pd.DataFrame(all_moves_4h)
        median_move_4h = moves_df.median(axis=1)

        # 5. Portfolio simulation
        print("Running simulation...")
        n_bars = len(unified_index)
        portfolio_equity = np.full(n_bars, float(initial_capital))
        curr_equity = float(initial_capital)

        # Track active positions: {symbol: {direction, entry_price, tp_price, sl_price, capital_allocated}}
        active_positions = {}
        all_trade_returns = []
        long_returns = []
        short_returns = []
        per_coin_trades = {s: [] for s in raw_data}

        # Warmup period: need enough bars for move lookback + EMA
        warmup = move_lookback + max(ema_fast, ema_slow) + 10

        for i in range(warmup, n_bars):
            # --- CHECK TP/SL ON EXISTING POSITIONS ---
            closed_symbols = []
            for symbol, pos in list(active_positions.items()):
                ind = coin_indicators[symbol]
                high_val = ind['high'].iloc[i]
                low_val = ind['low'].iloc[i]
                close_val = ind['close'].iloc[i]

                if pd.isna(close_val):
                    continue

                hit_tp = False
                hit_sl = False
                exit_price = close_val

                if pos['direction'] == 1:  # long
                    if high_val >= pos['tp_price']:
                        hit_tp = True
                        exit_price = pos['tp_price']
                    elif low_val <= pos['sl_price']:
                        hit_sl = True
                        exit_price = pos['sl_price']
                else:  # short
                    if low_val <= pos['tp_price']:
                        hit_tp = True
                        exit_price = pos['tp_price']
                    elif high_val >= pos['sl_price']:
                        hit_sl = True
                        exit_price = pos['sl_price']

                if hit_tp or hit_sl:
                    if pos['direction'] == 1:
                        ret = (exit_price - pos['entry_price']) / pos['entry_price']
                    else:
                        ret = (pos['entry_price'] - exit_price) / pos['entry_price']

                    ret -= 2 * fee_per_trade
                    pnl = pos['capital_allocated'] * ret
                    curr_equity += pnl

                    all_trade_returns.append(ret)
                    per_coin_trades[symbol].append(ret)
                    if pos['direction'] == 1:
                        long_returns.append(ret)
                    else:
                        short_returns.append(ret)

                    closed_symbols.append(symbol)

            for s in closed_symbols:
                del active_positions[s]

            # --- SCAN FOR NEW ENTRIES ---
            if len(active_positions) < max_trades:
                med_move = median_move_4h.iloc[i]

                if pd.isna(med_move) or abs(med_move) < move_threshold:
                    portfolio_equity[i] = curr_equity
                    continue

                # Strong directional move detected (median of all coins)
                is_bullish_move = med_move > 0

                # Collect all coin moves at this bar and compute lag Z-scores
                coin_moves = {}
                coin_meta = {}
                for symbol in raw_data:
                    if symbol in active_positions:
                        continue

                    ind = coin_indicators[symbol]
                    coin_move = ind['move_4h'].iloc[i]
                    close_val = ind['close'].iloc[i]
                    ema_bull = ind['ema_bullish'].iloc[i]

                    if pd.isna(coin_move) or pd.isna(close_val):
                        continue

                    coin_moves[symbol] = coin_move
                    coin_meta[symbol] = {'close': close_val, 'ema_bull': ema_bull}

                if len(coin_moves) < 3:
                    portfolio_equity[i] = curr_equity
                    continue

                # Compute lag = median - coin_move, then Z-score across all coins
                moves_arr = np.array(list(coin_moves.values()))
                lags = med_move - moves_arr
                lag_mean = np.mean(lags)
                lag_std = np.std(lags)

                if lag_std < 1e-10:
                    portfolio_equity[i] = curr_equity
                    continue

                # Filter coins by Z-score threshold and EMA
                lag_scores = {}
                symbols_list = list(coin_moves.keys())
                for j, symbol in enumerate(symbols_list):
                    z = (lags[j] - lag_mean) / lag_std
                    ema_bull = coin_meta[symbol]['ema_bull']

                    if is_bullish_move:
                        # Coin lagging upward: lag > 0, Z > threshold
                        if z >= lag_zscore and ema_bull:
                            lag_scores[symbol] = z
                    else:
                        # Coin lagging downward: lag < 0, Z < -threshold
                        if z <= -lag_zscore and not ema_bull:
                            lag_scores[symbol] = z

                if not lag_scores:
                    portfolio_equity[i] = curr_equity
                    continue

                # Pick the most extreme Z-score
                if is_bullish_move:
                    best_symbol = max(lag_scores, key=lag_scores.get)
                else:
                    best_symbol = min(lag_scores, key=lag_scores.get)

                if len(active_positions) >= max_trades:
                    portfolio_equity[i] = curr_equity
                    continue

                ind = coin_indicators[best_symbol]
                close_val = coin_meta[best_symbol]['close']
                coin_move = coin_moves[best_symbol]

                if is_bullish_move:
                    # LONG: coin should catch up to the median
                    direction = 1
                    remaining_pct = med_move - coin_move
                    tp_price = close_val * (1 + remaining_pct)
                    # SL at recent swing low
                    sl_price = ind['swing_low'].iloc[i]
                    if sl_price >= close_val:
                        sl_price = close_val * 0.98
                else:
                    # SHORT: coin should catch down to the median
                    direction = -1
                    remaining_pct = coin_move - med_move
                    tp_price = close_val * (1 - remaining_pct)
                    # SL at recent swing high
                    sl_price = ind['swing_high'].iloc[i]
                    if sl_price <= close_val:
                        sl_price = close_val * 1.02

                # Sanity checks
                if direction == 1 and tp_price <= close_val:
                    portfolio_equity[i] = curr_equity
                    continue
                if direction == -1 and tp_price >= close_val:
                    portfolio_equity[i] = curr_equity
                    continue

                # Allocate capital per trade
                capital_for_trade = curr_equity * exposure / max_trades

                active_positions[best_symbol] = {
                    'direction': direction,
                    'entry_price': close_val,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'capital_allocated': capital_for_trade,
                }

            portfolio_equity[i] = curr_equity

        # Close any remaining open positions at last bar
        for symbol, pos in active_positions.items():
            close_val = coin_indicators[symbol]['close'].iloc[-1]
            if pd.isna(close_val):
                continue
            if pos['direction'] == 1:
                ret = (close_val - pos['entry_price']) / pos['entry_price']
            else:
                ret = (pos['entry_price'] - close_val) / pos['entry_price']
            ret -= 2 * fee_per_trade
            all_trade_returns.append(ret)
            per_coin_trades[symbol].append(ret)
            if pos['direction'] == 1:
                long_returns.append(ret)
            else:
                short_returns.append(ret)
            pnl = pos['capital_allocated'] * ret
            curr_equity += pnl
        portfolio_equity[-1] = curr_equity

        # 6. Build portfolio DataFrame
        portfolio_df = pd.DataFrame({
            'equity': portfolio_equity
        }, index=unified_index)
        portfolio_df['drawdown'] = portfolio_df['equity'] / portfolio_df['equity'].cummax() - 1

        # 7. Compute metrics
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

            avg_rr = float(np.mean(trade_returns_arr[trade_returns_arr > 0])) / abs(float(np.mean(trade_returns_arr[trade_returns_arr < 0]))) if np.any(trade_returns_arr < 0) and np.any(trade_returns_arr > 0) else 0
        else:
            ev = win_rate = sharpe = omega = avg_rr = 0.0
            max_lose_streak = 0

        # 8. Print results
        print("\n" + "=" * 60)
        print("PER-COIN BREAKDOWN")
        print("=" * 60)
        for symbol in symbols:
            trades = per_coin_trades.get(symbol, [])
            if trades:
                coin_ev = np.mean(trades)
                coin_wr = np.mean(np.array(trades) > 0)
                print(f"  {symbol:12s} | Trades: {len(trades):4d} | EV: {coin_ev:+.2%} | WR: {coin_wr:.2%}")
            else:
                print(f"  {symbol:12s} | Trades:    0")

        print("\n" + "=" * 60)
        print("STAT-ARB PORTFOLIO RESULTS")
        print("=" * 60)
        print(f"  Return: {total_return:+.2%} | Max DD: {max_dd:.2%} | CAGR: {cagr:+.2%}")
        ev_long  = float(np.mean(long_returns))  if long_returns  else 0.0
        ev_short = float(np.mean(short_returns)) if short_returns else 0.0
        print(f"  Trades: {total_trades} | EV: {ev:+.2%} | Win Rate: {win_rate:.2%} | Avg R:R: {avg_rr:.2f}")
        print(f"  EV Long: {ev_long:+.2%} ({len(long_returns)} trades) | EV Short: {ev_short:+.2%} ({len(short_returns)} trades)")
        print(f"  Sharpe: {sharpe:.2f} | Omega: {omega:.2f} | Lose Streak: {max_lose_streak}")
        print(f"  Exposure/trade: {exposure:.0%}/{max_trades} | Fee: {fee_per_trade:.4%}/side")
        print("=" * 60)

        # 9. Plot
        if n_bars > 1:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(portfolio_df.index, portfolio_df['equity'].values,
                    linewidth=2, color='#2962FF', alpha=0.8)
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(True, which='both', alpha=0.2, linestyle='--')
            ax.grid(True, which='major', alpha=0.4)
            ax.set_title(f'Stat-Arb Equity | median benchmark | {len(symbols)} coins',
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity (log $)', fontsize=12)

            final_eq = portfolio_df['equity'].iloc[-1]
            ax.text(0.02, 0.98,
                f'Final: ${final_eq:,.0f}\nReturn: {total_return:+.1%}\nMax DD: {max_dd:.1%}\nTrades: {total_trades}',
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
            'avg_rr': avg_rr,
            'per_coin_trades': per_coin_trades,
        }


if __name__ == "__main__":
    bt = Backtester()

    symbols = [
        'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
        'DOT/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT',
        'SUI/USDT', 'ENA/USDT', 'XLM/USDT', 'AVAX/USDT','NEAR/USDT','APT/USDT',
    ]

    results = bt.backtest_stat_arb(
        symbols=symbols,
        timeframe='5m',
        move_threshold=0.04,        # 1.5% median move triggers scan
        move_lookback=24,           # 24 x 5m = 2h
        ema_fast=20,
        ema_slow=40,
        max_trades=2,
        swing_lookback=6,          # 30min for swing high/low
        lag_zscore=1.5,            # min Z-score of lag to qualify
        exposure=0.5,
        initial_capital=10000,
        fee_per_trade=0.0005,
    )
