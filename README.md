# S&P 500 Market Direction & Trend Predictor

**Team 5 — Afternoon Class (G2)**  
*EADA Business School — Global BBA & AI*

## Project Overview
This application is a Machine Learning tool built to predict short-term market directional trends (Up/Down) for the S&P 500 index (`^GSPC`). It fetches real-time financial market data, computes technical indicators, and trains a Random Forest Classifier to make directional predictions.

## Key Features
- **Data Pipeline:** Dynamic market data loading via `yfinance`.
- **Feature Engineering:** Technical indicators including Simple Moving Averages (20, 50, 200 SMA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Volatility.
- **Time-Series ML Pipeline:** Chronological train/test split preventing forward-looking bias.
- **Interactive UI:** Built with Streamlit and Plotly for real-time visualization and feature importance inspection.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY
   pip install -r requirements.txt
   streamlit run app.py
