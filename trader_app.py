# trader_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

from data_provider import get_price_history
from strategy import apply_sma_crossover
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh (or just copy helper)
from ai_models import add_direction_prediction

# ...

st.sidebar.subheader("🔄 Live Price Refresh")
enable_auto = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_secs = st.sidebar.slider("Refresh every (seconds)", 10, 300, 60)

if enable_auto:
    st_autorefresh(interval=refresh_secs * 1000, key="price_refresh")


st.set_page_config(page_title="Auto-Trading AI (Paper)", page_icon="📈")

st.title("📈 Auto-Trading AI — Multi-Ticker Strategy")
st.write("_Paper-trading demo — no real orders are sent._")

# -------------------------------------------------
# BASIC CONTROLS
# -------------------------------------------------
LISTED_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA"]
tickers = st.multiselect("Select ticker(s)", LISTED_TICKERS, default=["AAPL"])

period = st.selectbox("Data period", ["3mo", "6mo", "1y", "2y"], index=1)
short_window = st.slider("Short SMA window", 5, 30, value=10)
long_window = st.slider("Long SMA window", 20, 100, value=30)

# -------------------------------------------------
# ADVANCED CONTROLS
# -------------------------------------------------
st.subheader("⚙️ Strategy Controls")

col1, col2 = st.columns(2)

with col1:
    use_rsi_macd = st.checkbox("Require RSI + MACD confirmation", value=True)
    rsi_window = st.slider("RSI window", 7, 21, value=14)

with col2:
    use_vol_filter = st.checkbox("Apply volatility filter", value=False)
    vol_window = st.slider("Volatility lookback (days)", 5, 60, value=20)
    max_vol_pct = (
        st.slider("Max daily volatility (%)", 1.0, 10.0, value=5.0, step=0.5)
        if use_vol_filter
        else None
    )

use_ai = st.checkbox("Enable AI direction prediction", value=False)

st.subheader("🔐 Risk Management")

use_risk = st.checkbox("Enable Stop-Loss / Take-Profit", value=True)

stop_loss_pct = st.number_input(
    "Stop-Loss (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5
)

take_profit_pct = st.number_input(
    "Take-Profit (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5
)

st.subheader("💰 Portfolio & Costs")

col3, col4 = st.columns(2)

with col3:
    portfolio_capital = st.number_input(
        "Total portfolio capital (USD)", min_value=1000, value=10_000, step=1000
    )

with col4:
    trade_cost_bps = st.slider(
        "Per-trade cost (bps)", min_value=0, max_value=50, value=10, step=1
    )

# -------------------------------------------------
# HELPERS
# -------------------------------------------------


def preprocess_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Column normalization BEFORE indexing
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c).lower() for c in col if c]) for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Detect correct date column (Yahoo = 'date' after lowering)
    date_cols = [c for c in df.columns if c.startswith("date")]
    if not date_cols:
        raise KeyError("No date column found in DataFrame")

    df.rename(columns={date_cols[0]: "date"}, inplace=True)

    # Proper datetime conversion — FIXES 1970 issue permanently
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Detect close column
    close_cols = [c for c in df.columns if "close" in c]
    if not close_cols:
        raise KeyError("No close price column")
    df["close"] = df[close_cols[0]].astype(float)

    return df


def add_display_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD", "SL": "🛑 STOP-LOSS", "TP": "🎯 TAKE-PROFIT"}
    df["signal_display"] = df["signal"].map(emoji).fillna("⚪")
    df["date_display"] = df["date"].dt.strftime("%m-%d-%Y")
    return df


def calculate_metrics(df: pd.DataFrame, price_col: str):
    total_return_strategy = df["equity_curve"].iloc[-1] - 1
    total_return_bh = df[price_col].iloc[-1] / df[price_col].iloc[0] - 1

    trades = df[df["position"].diff() != 0]
    wins = trades[trades["strategy_returns_net"] > 0]
    win_rate = (len(wins) / len(trades) * 100) if len(trades) > 0 else 0.0

    return total_return_strategy, total_return_bh, win_rate, len(trades)


# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------

results: dict[str, tuple[pd.DataFrame, str]] = {}

if st.button("Run Strategy"):
    if not tickers:
        st.warning("Please select at least one ticker.")
    else:
        for ticker in tickers:
            st.header(f"📌 {ticker}")

            raw_df = get_price_history(ticker, period)
            if raw_df is None or raw_df.empty:
                st.error(f"No data returned for {ticker}")
                continue

            # Normalize OHLC and get clean 'date' + 'close'
            try:
                df = preprocess_ohlc(raw_df)
            except KeyError as e:
                st.error(f"{ticker}: {e}")
                continue

            # Apply strategy with indicators & trade costs
            try:
                df, price_col = apply_sma_crossover(
                    df,
                    short_window=short_window,
                    long_window=long_window,
                    use_rsi_macd=use_rsi_macd,
                    rsi_window=rsi_window,
                    use_vol_filter=use_vol_filter,
                    vol_window=vol_window,
                    max_vol_pct=max_vol_pct,
                    trade_cost_bps=trade_cost_bps,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    use_risk=use_risk,
                )
            except KeyError as e:
                st.error(f"{ticker}: Missing needed columns: {e}")
                continue
            
            # Apply AI model for direction prediction
            if use_ai:
                df = add_direction_prediction(df, price_col)
                # Display AI prediction
                last = df.iloc[-1]
                st.info(
                    f"AI prediction: **{last['pred_signal']}** "
                    f"(P(up)={last['pred_up_prob']:.2f}) for next bar."
                )

            # Baseline buy & hold equity curve
            df["bh_equity"] = (df[price_col] / df[price_col].iloc[0]).fillna(1.0)

            # Strategy equity curve is df["equity_curve"]


            # Add emoji display + formatted date
            df = add_display_enhancements(df)

            # Store for portfolio aggregation later
            results[ticker] = (df, price_col)

            # ----------------- PRICE + SMA CHART -----------------
            st.subheader("📈 Price, SMAs & Signals")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df["date"], df[price_col], label="Close", linewidth=1.3)
            ax.plot(df["date"], df["sma_short"], linestyle="--", label=f"SMA {short_window}")
            ax.plot(df["date"], df["sma_long"], linestyle="--", label=f"SMA {long_window}")

            # Buy/Sell markers (use raw 'signal' column, not display)
            buy_points = df[df["signal"] == "BUY"]
            sell_points = df[df["signal"] == "SELL"]

            ax.scatter(
                buy_points["date"],
                buy_points[price_col],
                marker="^",
                s=80,
                label="BUY",
            )
            ax.scatter(
                sell_points["date"],
                sell_points[price_col],
                marker="v",
                s=80,
                label="SELL",
            )

            # Risk management markers
            sl_points = df[df["signal"] == "SL"]
            tp_points = df[df["signal"] == "TP"]

            ax.scatter(sl_points["date"], sl_points[price_col], marker="X", color="red", s=120, label="Stop-Loss")
            ax.scatter(tp_points["date"], tp_points[price_col], marker="P", color="green", s=120, label="Take-Profit")

            ax.set_title(f"{ticker} — SMA Strategy with Signals")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)

            # ----------------- PERFORMANCE PANEL -----------------
            total_ret_strat, total_ret_bh, win_rate, n_trades = calculate_metrics(df, price_col)

            st.subheader("📊 Performance Summary")
            st.write(
                f"""
**Strategy Return (net of costs):** {total_ret_strat*100:.2f}%  
**Buy & Hold Return:** {total_ret_bh*100:.2f}%  
**Win-Rate:** {win_rate:.1f}%  
**Number of Trades:** {n_trades}  
**Trade Cost:** {trade_cost_bps} bps / trade
                """
            )

            if total_ret_strat > total_ret_bh:
                st.success("Strategy outperformed Buy & Hold 🚀")
            else:
                st.warning("Underperformed vs Buy & Hold 📉")

            # ----------------- INDICATOR SNAPSHOT -----------------
            st.subheader("📐 Latest Indicator Snapshot")
            last = df.iloc[-1]
            if use_rsi_macd:
                st.write(
                    f"RSI ({rsi_window}): **{last.get('rsi', float('nan')):.1f}** | "
                    f"MACD: **{last.get('macd', float('nan')):.4f}** | "
                    f"Signal: **{last.get('macd_signal', float('nan')):.4f}**"
                )
            if use_vol_filter and max_vol_pct is not None and "volatility" in df.columns:
                st.write(
                    f"Rolling volatility ({vol_window}d): "
                    f"**{last['volatility']*100:.2f}%** (max allowed {max_vol_pct:.1f}%)"
                )

            # ----------------- RECENT SIGNALS TABLE -----------------
            st.subheader("📅 Recent Trade Signals")
            sig_df = df[["date_display", price_col, "sma_short", "sma_long", "signal_display"]].tail(15)
            sig_df = sig_df.rename(
                columns={
                    "date_display": "Date",
                    price_col: "Close",
                    "sma_short": f"SMA {short_window}",
                    "sma_long": f"SMA {long_window}",
                    "signal_display": "Signal",
                }
            ).reset_index(drop=True)
            st.dataframe(sig_df, width=750)

            # ----------------- CSV EXPORT -----------------
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Export Full Strategy Data",
                data=csv,
                file_name=f"{ticker}_strategy_data.csv",
                mime="text/csv",
            )

            st.markdown("---")

        # -------------------------------------------------
        # PORTFOLIO AGGREGATION (EQUAL WEIGHTED)
        # -------------------------------------------------
        if results:
            st.header("📦 Portfolio View (Equal-Weighted)")

            frames = []
            for ticker, (df, _) in results.items():
                tmp = df[["date", "equity_curve"]].copy()
                tmp["equity_value"] = (portfolio_capital / len(results)) * tmp["equity_curve"]
                tmp = tmp[["date", "equity_value"]].rename(
                    columns={"equity_value": f"{ticker}_equity"}
                )
                frames.append(tmp)

            if len(frames) > 1:
                portfolio_df = reduce(
                    lambda left, right: pd.merge(left, right, on="date", how="outer"),
                    frames,
                )
            else:
                portfolio_df = frames[0]

            portfolio_df = portfolio_df.sort_values("date")
            equity_cols = [c for c in portfolio_df.columns if c.endswith("_equity")]
            portfolio_df[equity_cols] = portfolio_df[equity_cols].ffill()
            portfolio_df["total_equity"] = portfolio_df[equity_cols].sum(axis=1)

            total_return_portfolio = portfolio_df["total_equity"].iloc[-1] / portfolio_capital - 1

            st.subheader("📈 Portfolio Equity Curve")
            fig, ax = plt.subplots()
            ax.plot(portfolio_df["date"], portfolio_df["total_equity"])
            ax.set_title("Total Portfolio Equity (Equal-Weighted)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity (USD)")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            st.pyplot(fig)

            st.write(f"**Total Portfolio Return:** {total_return_portfolio*100:.2f}%")
