import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

# --- Page Setup ---
st.set_page_config(
    page_title="S&P 500 AI Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 S&P 500 Market Trend & Direction Predictor")
st.markdown("An interactive Machine Learning application designed for Team 5's AI Coding Project.")

# --- Sidebar Controls ---
st.sidebar.header("Model & Data Controls")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", datetime.now())

n_estimators = st.sidebar.slider("Model Estimators (Trees)", min_value=50, max_value=300, value=100, step=25)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=10, max_value=200, value=50, step=10)

# --- Data Loading & Feature Engineering ---
@st.cache_data(ttl=3600)
def load_and_preprocess_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    # 1. Target Definition (1 if Tomorrow's Close > Today's Close, else 0)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    # 2. Technical Indicators (Feature Engineering)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    
    # RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Volatility & Returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()
    
    df = df.dropna()
    return df

with st.spinner("Fetching market data and computing features..."):
    data = load_and_preprocess_data(ticker_symbol, start_date, end_date)

# Feature columns
feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "SMA_200", "RSI", "MACD", "Signal_Line", "Volatility"]

# --- Train ML Model ---
X = data[feature_cols]
y = data["Target"]

# Time-series train-test split (avoiding lookahead bias)
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier(
    n_estimators=n_estimators,
    min_samples_split=min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

# --- Top Dashboard Cards ---
col1, col2, col3, col4 = st.columns(4)
latest_close = data["Close"].iloc[-1]
latest_rsi = data["RSI"].iloc[-1]
latest_prob = y_prob[-1] * 100
next_day_pred = "UP 🟢" if y_pred[-1] == 1 else "DOWN 🔴"

col1.metric("Latest Close Price", f"${latest_close:,.2f}")
col2.metric("Latest RSI (14)", f"{latest_rsi:.2f}")
col3.metric("Next Trading Day Prediction", next_day_pred)
col4.metric("Bullish Confidence Probability", f"{latest_prob:.1f}%")

st.markdown("---")

# --- Visualizations ---
tab1, tab2, tab3 = st.tabs(["📊 Price & Technicals", "🤖 Model Performance", "🔍 Feature Importance"])

with tab1:
    st.subheader("Historical Price & Technical Indicators")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close Price", line=dict(color="blue", width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="50-day SMA", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_200"], name="200-day SMA", line=dict(color="red", width=1)))
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Out-of-Sample Backtest Results")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Model Accuracy", f"{acc * 100:.2f}%")
    m_col2.metric("Model Precision", f"{prec * 100:.2f}%")
    
    st.write("**Model Classification Performance Summary:**")
    results_df = pd.DataFrame({
        "Actual Direction": ["Down (0)" if v == 0 else "Up (1)" for v in y_test],
        "Predicted Direction": ["Down (0)" if v == 0 else "Up (1)" for v in y_pred],
        "Bullish Probability": [f"{p*100:.1f}%" for p in y_prob]
    }, index=y_test.index)
    st.dataframe(results_df.tail(20), use_container_width=True)

with tab3:
    st.subheader("Feature Importance Analysis")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        marker=dict(color="royalblue")
    ))
    fig_imp.update_layout(height=450, xaxis_title="Importance Score", yaxis_title="Technical Feature")
    st.plotly_chart(fig_imp, use_container_width=True)
