import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go 
from datetime import datetime, timedelta
import io

st.set_page_config(layout="wide",page_title="Multi stock dashboard")

#optional imports
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

#app helpers
st.cache_data(ttl=3600) #cache downloaded data for one hour 

#downloads hsitorical data(Open,high,low,close,volume) and returns clean data with datetime index
def fetch_ticker_data(ticker: str, start:str, end:str )-> pd.DataFrame:
    df = yf.download (ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index=pd.to_datetime(df.index)
    df=df[['open','high','low','close','volumn']]
    df=df.sort_index()
    return df


def moving_avg(df: pd.DataFrame, windows=[50,200]):
    for w in windows:
        df[f"MA_{w}"]=df['close'].rolling(window=w).mean()
    return df
    

