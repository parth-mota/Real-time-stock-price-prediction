from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def get_market_news(symbol):
    """recent news for a stock symbol using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return []

        processed_news = []
        for item in news[:5]:  # Get latest 5 news items
            processed_news.append({
                "title": item.get('title', ''),
                "publisher": item.get('publisher', ''),
                "link": item.get('link', ''),
                "published": datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return processed_news
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def get_stock_analysis(symbol):
    """technical analysis indicators for a stock"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['Signal'] = calculate_macd(hist['Close'])
        
        current_price = hist['Close'].iloc[-1]
        sma_20 = hist['SMA_20'].iloc[-1]
        sma_50 = hist['SMA_50'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        macd = hist['MACD'].iloc[-1]
        signal = hist['Signal'].iloc[-1]
        
        analysis = {
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "rsi": round(rsi, 2),
            "macd": round(macd, 2),
            "signal": round(signal, 2),
            "trend": get_trend_analysis(current_price, sma_20, sma_50, rsi, macd, signal)
        }
        
        return analysis
    except Exception as e:
        print(f"Error calculating technical analysis: {str(e)}")
        return {}

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate MACD technical indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def get_trend_analysis(price, sma20, sma50, rsi, macd, signal):
    """Analyze overall trend based on technical indicators"""
    signals = []
    
    # Trend based on moving averages
    if price > sma20 and sma20 > sma50:
        signals.append("Strong Uptrend")
    elif price < sma20 and sma20 < sma50:
        signals.append("Strong Downtrend")
    elif price > sma20:
        signals.append("Short-term Bullish")
    else:
        signals.append("Short-term Bearish")
    
    # RSI analysis
    if rsi > 70:
        signals.append("Overbought")
    elif rsi < 30:
        signals.append("Oversold")
    
    # MACD analysis
    if macd > signal:
        signals.append("MACD Bullish")
    else:
        signals.append("MACD Bearish")
    
    return signals

def predict_stock_price(symbol, days_to_predict=7):
    try:
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return {"error": "No data found for the given symbol"}

        # Get company info
        try:
            info = stock.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            market_cap = info.get('marketCap', 0)
            market_cap_formatted = f"${market_cap/1e9:.2f}B" if market_cap else 'N/A'
            pe_ratio = info.get('forwardPE', 'N/A')
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield = f"{dividend_yield * 100:.2f}%"
            else:
                dividend_yield = 'N/A'
        except:
            company_name, sector, market_cap_formatted = symbol, 'N/A', 'N/A'
            pe_ratio, dividend_yield = 'N/A', 'N/A'

        # Get technical analysis
        technical_analysis = get_stock_analysis(symbol)
        
        # Get recent news
        news = get_market_news(symbol)

        # Prepare data for prediction models
        sequence_length = 10
        X_train, X_test, y_train, y_test, X_train_rf, X_test_rf, scaler, scaled_data = prepare_data(df, sequence_length)

        # Train and get predictions from all models
        predictions, ensemble_metrics = train_and_predict_models(
            X_train, X_test, y_train, y_test, X_train_rf, X_test_rf,
            scaler, scaled_data, sequence_length, days_to_predict
        )

        # Calculate price changes
        current_price = float(df['Close'].iloc[-1])
        daily_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
        daily_change_pct = float((daily_change / df['Close'].iloc[-2]) * 100)

        # Get volume data
        volume_data = [int(x) for x in df['Volume'][-30:].values]
        avg_volume = int(np.mean(volume_data))
        
        return {
            "success": True,
            "company_info": {
                "name": company_name,
                "symbol": symbol.upper(),
                "sector": sector,
                "market_cap": market_cap_formatted,
                "pe_ratio": pe_ratio,
                "dividend_yield": dividend_yield
            },
            "technical_analysis": technical_analysis,
            "ensemble_metrics": ensemble_metrics,
            "current_price": round(current_price, 2),
            "daily_change": round(daily_change, 2),
            "daily_change_pct": round(daily_change_pct, 2),
            "predictions": predictions,
            "historical_dates": df.index[-30:].strftime('%Y-%m-%d').tolist(),
            "historical_values": [round(float(x), 2) for x in df['Close'][-30:].values],
            "volume_data": volume_data,
            "average_volume": avg_volume,
            "future_dates": [(df.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                           for i in range(days_to_predict)],
            "news": news
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

def prepare_data(df, sequence_length):
    """Prepare data for all models"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, X_test, y_train, y_test, X_train_rf, X_test_rf, scaler, scaled_data

def train_and_predict_models(X_train, X_test, y_train, y_test, X_train_rf, X_test_rf,
                           scaler, scaled_data, sequence_length, days_to_predict):
    
    # Train LSTM model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Train SVM model
    svm_model = SVR(kernel='rbf', C=100, gamma='auto')
    svm_model.fit(X_train_rf, y_train.ravel())

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train.ravel())

    # Generate predictions
    lstm_future, svm_future, rf_future, ensemble_future = [], [], [], []
    last_sequence = scaled_data[-sequence_length:]
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        # LSTM prediction
        lstm_next = lstm_model.predict(current_sequence.reshape(1, sequence_length, 1))[0, 0]
        lstm_next = float(scaler.inverse_transform([[lstm_next]])[0, 0])
        lstm_future.append(lstm_next)
        
        # SVM prediction
        svm_next = svm_model.predict(current_sequence.reshape(1, -1))[0]
        svm_next = float(scaler.inverse_transform([[svm_next]])[0, 0])
        svm_future.append(svm_next)
        
        # RF prediction
        rf_next = rf_model.predict(current_sequence.reshape(1, -1))[0]
        rf_next = float(scaler.inverse_transform([[rf_next]])[0, 0])
        rf_future.append(rf_next)
        
        # Calculate ensemble prediction (weighted average based on RMSE)
        ensemble_next = (lstm_next + svm_next + rf_next) / 3
        ensemble_future.append(ensemble_next)
        
        # Update sequence using ensemble prediction
        new_scaled_value = scaler.transform([[ensemble_next]])[0, 0]
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = new_scaled_value

    # Calculate model performance
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    lstm_test_pred = lstm_model.predict(X_test)
    lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
    lstm_rmse = float(calculate_percentage_rmse(y_test_actual, lstm_test_pred))

    svm_test_pred = svm_model.predict(X_test_rf)
    svm_test_pred = scaler.inverse_transform(svm_test_pred.reshape(-1, 1))
    svm_rmse = float(calculate_percentage_rmse(y_test_actual, svm_test_pred))

    rf_test_pred = rf_model.predict(X_test_rf)
    rf_test_pred = scaler.inverse_transform(rf_test_pred.reshape(-1, 1))
    rf_rmse = float(calculate_percentage_rmse(y_test_actual, rf_test_pred))

    # Calculate ensemble test predictions and RMSE
    ensemble_test_pred = (lstm_test_pred + svm_test_pred + rf_test_pred) / 3
    ensemble_rmse = float(calculate_percentage_rmse(y_test_actual, ensemble_test_pred))
    ensemble_r2 = r2_score(y_test_actual, ensemble_test_pred)
    ensemble_precision = r2_score(y_test_actual, ensemble_test_pred, multioutput='variance_weighted')
    ensemble_confidence = 1 - ensemble_rmse / np.mean(y_test_actual)

    return {
        "lstm": [round(x, 2) for x in lstm_future],
        "svm": [round(x, 2) for x in svm_future],
        "rf": [round(x, 2) for x in rf_future],
        "ensemble": [round(x, 2) for x in ensemble_future],
    }, {
        "ensemble_rmse": round(ensemble_rmse, 2),
        "ensemble_r2": round(ensemble_r2, 2),
        "ensemble_precision": round(ensemble_precision, 2),
        "ensemble_confidence": round(ensemble_confidence, 2)
    }

def calculate_percentage_rmse(y_true, y_pred):
    """Calculate RMSE as a percentage of the mean price"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_price = float(np.mean(y_true))
    return (rmse / mean_price) * 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form.get('symbol', '')
    days = int(request.form.get('days', 7))
    return jsonify(predict_stock_price(symbol, days))

if __name__ == '__main__':
    app.run(debug=True)