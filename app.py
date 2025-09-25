import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from io import StringIO
import requests
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


# Load Nasdaq tickers
@st.cache_data(ttl=86400)
def load_nasdaq_tickers():
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to download Nasdaq tickers: {response.status_code}")
        return []
    data = StringIO(response.text)
    df = pd.read_csv(data, sep="|")
    df = df[:-1]  # remove footer
    return df['Symbol'].tolist()

nasdaq_tickers = load_nasdaq_tickers()


def flatten_columns(cols):
    
    def elem_to_list(elem):
        # Recursively convert nested tuple/list into list of strings
        if isinstance(elem, (tuple, list)):
            out = []
            for e in elem:
                out.extend(elem_to_list(e))
            return out
        else:
            return [str(elem)]

    flat = []
    for col in cols:
        if isinstance(col, (tuple, list)):
            parts = [p for p in elem_to_list(col) if p not in [None,""]]
            flat.append("_".join(parts))
        else:
            flat.append(str(col))
    return flat

# app helpers
@st.cache_data(ttl=3600)  # cache downloaded data for one hour

# downloads hsitorical data(Open,high,low,close,volume) and returns clean data with datetime index
def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, 
                     start=start, 
                     end=end,
                     progress=False,
                     auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    
#to handle Multiindex
    if isinstance(df.columns,pd.MultiIndex):
        if 'Ticker' in df.columns.names:
            df.columns=df.columns.droplevel('Ticker')
        else:
            #flatten multiindex to string(fallback)
           df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]



#normalise column names
    rename_map = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
        # handle lowercase
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    #keep standard OHLCV if present 
    keep_cols=[c for c in ['Open','High','Low','Close','Volume']if c in df.columns]
    df=df[keep_cols]

    df.index = pd.to_datetime(df.index)
    df = df[['Open','High','Low','Close','Volume']]
    df = df.sort_index()
    return df


def moving_avg(df: pd.DataFrame, windows=[50, 200]):
    for w in windows:
        df[f"MA_{w}"] = df['Close'].rolling(window=w).mean()
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
    return pd.DataFrame({"ds": future_dates, 'yhat': yhat})




def prophet_forecast(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    from prophet import Prophet as ProphetModel

    if df is None or df.empty:
        raise ValueError("Input df is empty or None — cannot run Prophet.")

    df = df.rename(columns=lambda c: str(c).strip().capitalize())

    if "Close" not in df.columns:
        raise ValueError(f"'Close' not found. Columns are: {df.columns.tolist()}")

    df_prophet = df.reset_index()
    first_col = df_prophet.columns[0]
    df_prophet = df_prophet.rename(columns={first_col: "ds", "Close": "y"})
    df_prophet = df_prophet[["ds", "y"]]
    df_prophet["ds"]=pd.to_datetime(df_prophet["ds"])
    df_prophet["y"]=df_prophet["y"].astype(float)   #forces float

    m = ProphetModel(daily_seasonality=True)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=periods, freq="B")
    forecast = m.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]



# UI:sliders
# Text input for any ticker, comma-separated
st.sidebar.header("Select tickers")
tickers = st.sidebar.multiselect(
    "Choose one or more Nasdaq tickers",
    options=nasdaq_tickers,
    default=["AAPL", "MSFT"])

start_default = datetime.today() - timedelta(days=365)
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
st.title("Multi ticker stock analysis and Forecasting")
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
summary_rows=[]
for t, df in data_dict.items():
    if not df.empty:
     last_close=df['Close'].iloc[-1]
     first_close=df['Close'].iloc[0]
     pct_change=(last_close/first_close-1)*100
     vol=int(df['Volume'].iloc[-1])

else:
    vol=None
    pct_change=None
    last_close=None
    summary_rows.append({
        'Ticker': t,
        'last Close': round(last_close, 2) if last_close is not None else None,
        '% Change (range)': round(pct_change, 2) if pct_change is not None else None,
        'Last Volume': vol
    })
if summary_rows:
    summary_df=pd.DataFrame(summary_rows).set_index('Ticker')
    st.subheader('snapshot')
    st.table(summary_df)

#Display each ticker as expanable section 
import traceback

for t, df in data_dict.items():
    with st.expander(f"{t}-Chart and Forecast",expanded=False):
     col1, col2=st.columns([2,1])
     with col1:
         st.subheader(f"{t}price")
         fig=plot_candlestick_with_mas(df,t,ma_windows)
         st.plotly_chart(fig,use_container_width=True)
         if 'Volume' in df.columns:
           vol_fig=go.Figure([go.Bar(x=df.index,y=df['Volume'],name='Volume')])
           vol_fig.update_layout(height=200,margin=dict(t=10,b=10))
           st.plotly_chart(vol_fig,use_container_width=True)
         else:
           st.info("No Volume data avaliable for this ticker")
    
     
    with col2:
        st.subheader('Details')
        st.write(f"Latest Close: **{df['Close'].iloc[-1]:.2f}**")    
        st.write(f"Data Points:{len(df)}")
        if 'rsi_14' in df.columns:
            st.write(f"Latest RSI (14):{df['rsi_14'].iloc[-1]:2f}")


#download
if download_all:
    csv=df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(f"Download {t} CSV",csv,file_name=f"{t}_History.csv",mime="text/csv")



st.markdown("---")
st.subheader("Forecast")

if PROPHET_AVAILABLE:
    with st.spinner(f"fitting prophet for {t}..."):
        try:
            fcst=prophet_forecast(df.copy(),periods=forecast_days)
            #plot forcast
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],name='yhat'))
            fig2.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat_upper'],name='upper',line=dict(width=0),showlegend=False))
            fig2.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat_lower'],name='lower',line=dict(width=0),fill='tonexty',fillcolor='rgba(0,100,80,0.1)',showlegend=False))
            fig2.update_layout(title=f"{t} Prophet Forecast ({forecast_days} business days ahead)",height=400)
            st.plotly_chart(fig2,use_container_width=True)
            st.dataframe(fcst.tail(10).set_index("ds"))
        except Exception as e:
            st.error(f"Prophet failed for {t}:{e}")
            st.text(traceback.format_exc())
            st.info('using simple rolling forecast instead')
            fcst=simple_rolling_forecast(df,periods=forecast_days)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name='historical'))
            fig2.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast', line=dict(dash='dash')))
            fig2.update_layout(title=f"{t} Naive Forecast ({forecast_days} business days ahead)", height=400)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(fcst.set_index('ds').head(10))


else:
    with st.spinner("Using simple rollimg mean forcast(Prophet not installed)"):
        fcst=simple_rolling_forecast(df, periods=forecast_days)
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=df.index,y=df['Close'],name='historical'))
        fig2.add_trace(go.Scatter(x=fcst['ds'],y=fcst['yhat'],name='forecast',line=dict(dash='dash')))
        fig2.update_layout(title=f"{t} Navie Forecast({forecast_days}) business days ahead)", height=400)
        st.plotly_chart(fig2,use_container_width=True)
        st.dataframe(fcst.set_index("ds").head(10))

st.sidebar.markdown("**Tips:** for many tickers, consider fecthing data with 'yf.download(list_of_tickers)' for speed(yfinance supports batch downloads).")


