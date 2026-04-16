import pandas as pd
import numpy as np

def factorial(n: int) -> float:
    result = 1.0
    for i in range(1, n + 1):
        result *= i
    return result


def pascal_filter(length: int) -> list:
    n = length - 1
    weights, coeff = [], 1.0
    for k in range(n + 1):
        weights.append(coeff)
        coeff = coeff * (n - k) / (k + 1)
    return weights


def binomial_ma(src: pd.Series, coeffs: list) -> pd.Series:
    """Binomial moving average."""
    result   = pd.Series(index=src.index, dtype=float)
    n_coeffs = len(coeffs)
    for i in range(len(src)):
        if i < n_coeffs - 1:
            result.iloc[i] = src.iloc[i]
            continue
        s = w = 0.0
        for j in range(n_coeffs):
            s += src.iloc[i - j] * coeffs[j]
            w += coeffs[j]
        result.iloc[i] = s / w if w > 0 else src.iloc[i]
    return result


def atr_binomial(high: pd.Series, low: pd.Series,
                 close: pd.Series, coeffs: list) -> pd.Series:
    """ATR using binomial filter."""
    result   = pd.Series(index=close.index, dtype=float)
    n_coeffs = len(coeffs)
    for i in range(len(close)):
        if i < n_coeffs - 1:
            result.iloc[i] = high.iloc[i] - low.iloc[i]
            continue
        s = w = 0.0
        for j in range(n_coeffs):
            idx = i - j
            if idx == 0:
                tr = high.iloc[idx] - low.iloc[idx]
            else:
                tr = max(high.iloc[idx] - low.iloc[idx],
                         abs(high.iloc[idx] - close.iloc[idx - 1]),
                         abs(low.iloc[idx]  - close.iloc[idx - 1]))
            s += tr * coeffs[j]
            w += coeffs[j]
        result.iloc[i] = s / w if w > 0 else (high.iloc[i] - low.iloc[i])
    return result


def frama(df: pd.DataFrame, length: int = 7) -> pd.Series:
    """FRAMA matching PineScript ta.frama."""
    if len(df) < length:
        return pd.Series(np.nan, index=df.index)

    high  = df['high']
    low   = df['low']
    close = df['close']

    half  = max(1, length // 2)
    N3    = (high.rolling(length).max() - low.rolling(length).min()) / length
    N1    = (high.rolling(half).max()   - low.rolling(half).min())   / half
    N2    = (high.shift(half).rolling(half).max() -
             low.shift(half).rolling(half).min()) / half

    eps   = 1e-10
    ratio = np.clip((N1 + N2) / (N3 + eps), eps, None)
    d     = np.clip(np.log(ratio) / np.log(2), 0, 1)
    alpha = np.clip(np.exp(-4.6 * (d - 1)), 0.01, 1)

    result = [close.iloc[0]]
    for i in range(1, len(close)):
        a   = 2.0 / (length + 1) if i < length else alpha.iloc[i]
        result.append(a * close.iloc[i] + (1 - a) * result[-1])

    return pd.Series(result, index=close.index)


def quark(df: pd.DataFrame) -> pd.DataFrame:
    """Quark strategy (PineScript port)."""
    if len(df) < 100:
        df['signal'] = 0
        return df

    close = df['close']
    high  = df['high']
    low   = df['low']

    # 1. MADSD
    linreg = pd.Series(index=close.index, dtype=float)
    lin_len, offset = 60, 10
    for i in range(len(close)):
        if i >= lin_len + offset - 1:
            w   = close.iloc[i - (lin_len + offset) + 1 : i - offset + 1].values
            c   = np.polyfit(np.arange(len(w)), w, 1)
            linreg.iloc[i] = c[0] * (len(w) - 1) + c[1]
        else:
            linreg.iloc[i] = close.iloc[i]

    MAD_val = close.rolling(25).apply(
        lambda x: np.abs(x - np.median(x)).mean(), raw=True)
    upper, lower = linreg + MAD_val, linreg - MAD_val

    MADSD = pd.Series(0, index=close.index)
    for i in range(len(MADSD)):
        if close.iloc[i] > upper.iloc[i]:
            MADSD.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i]:
            MADSD.iloc[i] = -1
        elif i > 0:
            MADSD.iloc[i] = MADSD.iloc[i - 1]

    # 2. PFATR
    coeffs = pascal_filter(13)
    bima   = binomial_ma(close, coeffs)
    bimatr = atr_binomial(high, low, close, coeffs)
    thr_l  = bima + 2.6 * bimatr
    thr_s  = bima - 2.0 * bimatr

    PFATR = pd.Series(0, index=close.index)
    for i in range(len(PFATR)):
        if close.iloc[i] > thr_l.iloc[i]:
            PFATR.iloc[i] = 1
        elif close.iloc[i] < thr_s.iloc[i]:
            PFATR.iloc[i] = -1
        elif i > 0:
            PFATR.iloc[i] = PFATR.iloc[i - 1]

    # 3. ZFR (FRAMA Z-Score)
    fr    = frama(df, 7)
    mean  = fr.rolling(29).mean()
    std   = fr.rolling(29).std()
    zscore = (fr - mean) / (std + 1e-10)

    ZFR = pd.Series(0, index=close.index)
    for i in range(len(ZFR)):
        if zscore.iloc[i] > 1:
            ZFR.iloc[i] = 1
        elif zscore.iloc[i] < 0:
            ZFR.iloc[i] = -1
        elif i > 0:
            ZFR.iloc[i] = ZFR.iloc[i - 1]

    # 4. Combine
    TPI    = pd.DataFrame({'MADSD': MADSD, 'PFATR': PFATR, 'ZFR': ZFR},
                          dtype=float).mean(axis=1)
    signal = pd.Series(0, index=close.index)
    for i in range(len(signal)):
        if TPI.iloc[i] > 0.1:
            signal.iloc[i] = 1
        elif TPI.iloc[i] < -0.1:
            signal.iloc[i] = -1
        elif i > 0:
            signal.iloc[i] = signal.iloc[i - 1]

    df['signal'] = signal
    return df