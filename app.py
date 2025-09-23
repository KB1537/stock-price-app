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
@st.cache_data(ttl=3600)  # cache downloaded data for one hour

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
    future_dates = pd.date_range(
        # B for business days
        start=last_date + pd.Timedelta(days=1), periods=periods, freq='B')
    # uses mean of last 30 closes and yhat
    yhat = [df['Close'].tail(30).mean()]*periods
    return pd.DataFrame({'ds': future_dates, 'yhat': yhat})


def prophet_forcast(df: pd.DataFrame, periods: int = 90):
    df_prophet=df.reset_index().rename(
        columns={'index', 'ds', 'Close', 'y'})[['ds', 'y']]
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods, freq='B')
    forcast = m.predict(future)
    return forcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# UI:sliders

st.sidebar.header('Header')
default_tickers = ['APPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
tickers = st.sidebar.multiselect(
    "Select ticker(multi)", options=default_tickers, default=['AAPL', 'MSFT'])

start_default = datetime.today - timedelta(days=365)
start_date = st.sidebar.date_input('start date', start_default)
end_date = st.sidebar.date_input('end date', datetime.today())

forecast_days = st.sidebar.slider(
    "Forecast horizon (business days)", min_value=5, max_value=365, value=90)
ma_windows = st.sidebar.multiselect("Show moving averages", options=[
                                    20, 50, 100, 200], default=[50, 200])
add_technical = st.sidebar.checkbox(
    "Add technical indicators (pandas_ta)", value=False)
download_all = st.sidebar.checkbox("Allow CSV download per ticker", value=True)

st.sidebar.markdown("---")
if PROPHET_AVAILABLE:
    st.sidebar.success("Prophet available for forecasting")
else:
    st.sidebar.info("Prophet not installed — fallback forecast will be used")


#Main code for app
st.title("Multi ticker stock analysis and forcasting")
st.markdown("Select multiple tickers on the left, then scroll through each ticker's chart, indicators, and forecast.")

if not tickers:
    st.warning("Pick at least one ticker from the sidebar to begin.")
    st.stop()

#fetch data(one ticker at a time)

data_dict={}
with st.spinner("Downlading data"):
    for t in tickers:
        df=fetch_ticker_data(t, start_date.isoformat(), end_date.isoformat())
        if df.empty:
            st.error(f"No data found for {t}-check ticker symbol or date range.")
        else:
            df=moving_avg(df,ma_windows)
            if add_technical and PANDAS_TA_AVAILABLE:
                df['rsi_14']=pta.rsi(df['Close'],lenght=14)
                macd = pta.macd(df['Close'])
                df=pd.concat([df, macd],axis=1)
            data_dict[t]=df


# Summary table 
summary+rows=[]
for t, df in data_dict.items():
    last_close=df['Close']