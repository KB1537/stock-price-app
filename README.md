# Multi-Ticker Stock Analysis & Forecasting Dashboard

A Streamlit application to visualize stock prices, technical indicators, and forecasts for multiple Nasdaq-listed tickers using yfinance, plotly, and optional Prophet forecasting.

# Table of Contents

**Features**

1. [Features](#features)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [How It Works](#how-it-works)  
5. [Forecasting](#forecasting)  
6. [Technical Indicators](#technical-indicators)  
7. [Screenshots](#screenshots)  
8. [Tips & Notes](#tips--notes)  

# Features

- Fetch historical stock data (Open, High, Low, Close, Volume) from Yahoo Finance.

- Supports any Nasdaq-listed ticker via a searchable multiselect.

- Candlestick charts with customizable moving averages (20, 50, 100, 200).

- Optional technical indicators: RSI, MACD (requires pandas_ta).

- Forecast future prices using Prophet (if installed) or simple rolling mean as fallback.

- Download CSV of historical data for each ticker.

- Business-day-aware forecasts.

# Installation

**Clone Repo**

**git clone** https://github.com/KB1537/stock-price-app/blob/main/app.py


**Create a Python virtual environment (optional but recommended):**

- python -m venv .venv
     - source .venv/bin/activate for Linux/macOS
     - .venv\Scripts\activate for Windows 


**Install dependencies:**

- streamlit 
- yfinance 
- plotly 
- pandas


Optional packages:

- prophet 
- pandas_ta

# Usage

**Run the app locally**:

streamlit run app.py


- Use the sidebar to select Nasdaq tickers.

- Choose date range, moving averages, and forecast horizon.

- Enable technical indicators and CSV downloads if needed.

# How It Works

**Ticker Selection**:

- Uses the official Nasdaq listing from nasdaqtrader.com for autocomplete.

- Users can select multiple tickers.

# Data Fetching:

- Fetches historical OHLCV data from Yahoo Finance (yfinance).

- Handles MultiIndex columns automatically.

# Charting:

- Candlestick charts using Plotly with overlaid moving averages.

- Volume charts displayed below price charts.

# Forecasting

- Prophet Forecast: Uses fbprophet (or prophet) if installed.

- Forecast horizon set by sidebar slider (5–365 business days).
- Shows predicted price, upper and lower bounds.

# Naive Forecast

- If Prophet is not installed or fails, a rolling mean forecast is used.

- Forecast line is dashed and displayed with historical data.

# Technical Indicators

- Optional indicators available if pandas_ta is installed:

- RSI (14-period) – Relative Strength Index

- MACD – Moving Average Convergence Divergence

# Tips & Notes

- For many tickers, consider batch downloading with yf.download(list_of_tickers) for faster performance.

- The app caches data for 1 hour to reduce repeated downloads.

- Ensure ticker symbols are valid Nasdaq symbols to avoid errors.