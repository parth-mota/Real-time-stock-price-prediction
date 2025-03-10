import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="ProTrader Insights", layout="wide")

# Custom CSS to style the app
st.markdown("""
<style>
    .reportview-container {
        background-color: #1e1e1e;
    }
    .main {
        background-color: #1e1e1e;
    }
    h1, h2, h3 {
        color: #90EE90;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        color: #1e1e1e;
        background-color: #90EE90;
        border-radius: 5px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        color: #90EE90;
        background-color: #2d2d2d;
        border-color: #90EE90;
    }
    p {
        color: #ffffff;
    }
    .stMetricValue {
        color: #90EE90;
    }
    .css-1k58x6s {
        border: 2px solid #90EE90;
        border-radius: 10px;
        padding: 10px;
        background-color: #2d2d2d;
    }
</style>
""", unsafe_allow_html=True)

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(data):
    data['Target'] = data['Close'].shift(-1)
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['ATR'] = calculate_atr(data)
    data.dropna(inplace=True)
    
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'RSI', 'MACD', 'ATR']]
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def train_and_predict(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(kernel='rbf'),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test.iloc[-1].values.reshape(1, -1))[0]
        accuracy = model.score(X_test, y_test)
        results[name] = {'prediction': prediction, 'accuracy': accuracy}
    
    return results

# Streamlit app
st.title('🚀 ProTrader Insights')

# User input
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL)', 'AAPL')
with col2:
    days = st.number_input('Number of days for historical data', min_value=30, value=365)

if st.button('Analyze and Predict'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get stock data
    data = get_stock_data(ticker, start_date, end_date)
    
    # Display current price
    current_price = data['Close'].iloc[-1]
    st.metric("Current Price", f"${current_price:.2f}", f"{data['Close'].pct_change().iloc[-1]:.2%}")
    
    # Prepare data and train models
    X_train, X_test, y_train, y_test = prepare_data(data)
    predictions = train_and_predict(X_train, X_test, y_train, y_test)
    
    # Display results
    st.subheader('Price Predictions and Model Accuracy')
    cols = st.columns(5)
    for idx, (model, result) in enumerate(predictions.items()):
        cols[idx].metric(
            model, 
            f"${result['prediction']:.2f}", 
            f"{(result['prediction'] - current_price) / current_price:.2%}",
            help=f"Model Accuracy: {result['accuracy']:.2%}"
        )
    
    # Determine best model
    best_model = max(predictions, key=lambda x: predictions[x]['accuracy'])
    st.success(f"🏆 Best performing model: {best_model} (Accuracy: {predictions[best_model]['accuracy']:.2%})")
    
    # Plot historical data with candlestick chart
    st.subheader('Historical Stock Price')
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='#90EE90')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot volume
    volume_fig = px.bar(data, x=data.index, y='Volume', title=f'{ticker} Trading Volume')
    volume_fig.update_traces(marker_color='#90EE90')
    volume_fig.update_layout(
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='#90EE90')
    )
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Plot technical indicators
    st.subheader('Technical Indicators')
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI plot
        rsi_fig = px.line(data, x=data.index, y='RSI', title='Relative Strength Index (RSI)')
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        rsi_fig.update_layout(
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#90EE90')
        )
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    with col2:
        # MACD plot
        macd_fig = px.line(data, x=data.index, y=['MACD', 'Signal_Line'], title='MACD')
        macd_fig.update_layout(
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#90EE90')
        )
        st.plotly_chart(macd_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Created by ProTrader Insights Team")
st.markdown("Disclaimer: This tool is for educational purposes only. Always conduct thorough research before making investment decisions.")