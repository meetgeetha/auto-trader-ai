import pandas as pd
import numpy as np

def _normalize_and_find_price_col(df: pd.DataFrame):
    df = df.copy()

    # Flatten / lowercase columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c).lower() for c in col if c]) for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    price_col = None
    for c in df.columns:
        if "close" in c:
            price_col = c
            break

    if price_col is None:
        raise KeyError(f"No close column found. Columns: {df.columns.tolist()}")

    df[price_col] = df[price_col].astype(float)
    return df, price_col


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def apply_sma_crossover(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
    use_rsi_macd: bool = False,
    rsi_window: int = 14,
    use_vol_filter: bool = False,
    vol_window: int = 20,
    max_vol_pct: float | None = None,
    trade_cost_bps: int = 0,
    stop_loss_pct: float | None = None,    # e.g. 5.0 for 5%
    take_profit_pct: float | None = None,  # e.g. 10.0 for 10%
    use_risk: bool = False,
    **kwargs
) -> tuple[pd.DataFrame, str]:
    df = df.copy()

    # ---- 1) Normalize columns, find close ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c).lower() for c in col if c) for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    price_col = None
    for c in df.columns:
        if "close" in c:
            price_col = c
            break
    if price_col is None:
        raise KeyError(f"No close column found. Columns: {df.columns.tolist()}")

    # ---- 2) Basic SMAs ----
    df["sma_short"] = df[price_col].rolling(short_window).mean()
    df["sma_long"] = df[price_col].rolling(long_window).mean()

    df["signal"] = "HOLD"
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = "BUY"
    df.loc[df["sma_short"] < df["sma_long"], "signal"] = "SELL"

    # ---- 3) Optional RSI + MACD ----
    if use_rsi_macd:
        df["rsi"] = _compute_rsi(df[price_col], rsi_window)
        macd, macd_signal, _ = _compute_macd(df[price_col])
        df["macd"] = macd
        df["macd_signal"] = macd_signal

        # require confirmation: only BUY if RSI low + MACD cross up, etc. (simple demo rule)
        buy_mask = (df["signal"] == "BUY") & (df["rsi"] < 60) & (df["macd"] > df["macd_signal"])
        sell_mask = (df["signal"] == "SELL") & (df["rsi"] > 40) & (df["macd"] < df["macd_signal"])

        df["signal"] = "HOLD"
        df.loc[buy_mask, "signal"] = "BUY"
        df.loc[sell_mask, "signal"] = "SELL"

    # ---- 4) Optional volatility filter ----
    if use_vol_filter and max_vol_pct is not None:
        df["returns_raw"] = df[price_col].pct_change()
        df["volatility"] = df["returns_raw"].rolling(vol_window).std()
        high_vol = df["volatility"] * 100 > max_vol_pct
        df.loc[high_vol, "signal"] = "HOLD"
    else:
        df["volatility"] = np.nan

    # ---- 5) Risk-controlled backtest: SL/TP + costs ----
    df["returns"] = df[price_col].pct_change().fillna(0.0)

    position = 0   # 1 = long, 0 = flat, -1 = short (we’ll keep it long/flat only for now)
    entry_price = None
    equity_curve = [1.0]
    positions = [0]
    strat_returns = []

    cost = trade_cost_bps / 10000.0  # e.g. 10 bps -> 0.001

    for i in range(1, len(df)):
        price_prev = df.iloc[i - 1][price_col]
        price_now = df.iloc[i][price_col]
        sig = df.iloc[i]["signal"]

        daily_ret = (price_now / price_prev) - 1.0 if price_prev else 0.0

        # default: carry current position
        new_position = position

        # entry logic (simple: act on BUY/SELL)
        if sig == "BUY" and position == 0:
            new_position = 1
            entry_price = price_now
            daily_ret -= cost  # pay entry cost
        elif sig == "SELL" and position == 1:
            new_position = 0
            entry_price = None
            daily_ret -= cost  # pay exit cost

        # risk controls: evaluate only if in position and risk management enabled
        if position == 1 and entry_price is not None and use_risk:
            move_from_entry = (price_now / entry_price) - 1.0

            hit_sl = (stop_loss_pct is not None) and (move_from_entry <= -stop_loss_pct / 100.0)
            hit_tp = (take_profit_pct is not None) and (move_from_entry >= take_profit_pct / 100.0)

            if hit_sl:
                new_position = 0
                entry_price = None
                daily_ret -= cost  # exit cost on forced close
                # Update signal to indicate stop loss
                df.at[df.index[i], "signal"] = "SL"
            elif hit_tp:
                new_position = 0
                entry_price = None
                daily_ret -= cost  # exit cost on forced close
                # Update signal to indicate take profit
                df.at[df.index[i], "signal"] = "TP"

        strat_ret = daily_ret * position  # use *previous* position
        last_equity = equity_curve[-1]
        new_equity = last_equity * (1.0 + strat_ret)

        equity_curve.append(new_equity)
        positions.append(new_position)
        strat_returns.append(strat_ret)

        position = new_position

    df["position"] = pd.Series(positions, index=df.index)
    
    # Calculate returns and strategy performance using vectorized approach
    df["returns"] = df[price_col].pct_change()
    df["strategy_returns"] = df["returns"] * df["position"]

    # Apply trade cost at transitions
    cost_factor = 1 - (trade_cost_bps / 10000)
    df.loc[df["position"].diff() != 0, "strategy_returns"] *= cost_factor

    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()

    # Net of costs already baked in
    df["strategy_returns_net"] = df["strategy_returns"]

    return df, price_col


# ----- helpers -----


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist