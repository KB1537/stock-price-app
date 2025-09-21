import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide", page_title="Multi stock dashboard")

# optional imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except Exception:
    PANDAS_TA_AVAILABLE = False

# app helpers
st.cache_data(ttl=3600)  # cache downloaded data for one hour

# downloads hsitorical data(Open,high,low,close,volume) and returns clean data with datetime index


def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'high', 'Low', 'Close', 'Volumn']]
    df = df.sort_index()
    return df


def moving_avg(df: pd.DataFrame, windows=[50, 200]):
    for w in windows:
        df[f"MA_{w}"] = df['close'].rolling(window=w).mean()
    return df


def plot_candlestick_with_mas(df: pd.DataFrame, ticker: str, ma_windows=[50, 200]):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f"{ticker} OHLC"
    ))
    for w in ma_windows:
        if f"MA_{w}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f"MA_{w}"], name=f"MA{w}", mode='lines', line=dict(width=1)))
    fig.update_layout(title=f"{ticker} — Candlestick + MAs",
                      xaxis_rangeslider_visible=False, height=450)
    return fig


def simple_rolling_forecast(df: pd.DataFrame, periods: int = 30):
    """Naive forecast: extend the last value or use last-window mean (simple fallback)."""
    last_date = df.index.max()
    last_val = df['Close'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1)periods=periods freq='B')
    # uses mean of last 30 closes and yhat
    yhat = [df['Close'].tail(30).mean()]*periods
    return pd.DataFrame({'ds': future_dates, 'yhat': yhat})

    