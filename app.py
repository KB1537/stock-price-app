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