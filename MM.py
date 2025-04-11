import yfinance as yf
import ta
import requests
import pandas as pd
from datetime import datetime, timedelta
import openpyxl
import numpy as np
import json
import os
from dl_model import get_dl_predictions

# Finnhub API Configuration
FINNHUB_API_KEY = "cvsepvpr01qhup0qc89gcvsepvpr01qhup0qc8a0"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Prediction tracking file
PREDICTION_HISTORY_FILE = 'prediction_history.json'

def load_prediction_history():
    if os.path.exists(PREDICTION_HISTORY_FILE):
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_prediction_history(history):
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def track_prediction(prediction_date, predictions, actual_returns):
    try:
        history = load_prediction_history()
        
        # Calculate prediction accuracy
        gainers_accuracy = calculate_accuracy(predictions['gainers'], actual_returns)
        losers_accuracy = calculate_accuracy(predictions['losers'], actual_returns)
        overall_accuracy = calculate_overall_accuracy(predictions, actual_returns)
        
        accuracy_metrics = {
            'gainers_accuracy': gainers_accuracy,
            'losers_accuracy': losers_accuracy,
            'overall_accuracy': overall_accuracy
        }
        
        # Store prediction data with proper Series handling
        prediction_record = {
            'date': prediction_date,
            'predictions': predictions,
            'actual_returns': {
                k: float(v.iloc[0] if isinstance(v, pd.Series) else v) 
                for k, v in actual_returns.items()
            },
            'accuracy_metrics': accuracy_metrics
        }
        
        history.append(prediction_record)
        save_prediction_history(history)
        
        return accuracy_metrics
        
    except Exception as e:
        print(f"Error in track_prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'gainers_accuracy': 0.0,
            'losers_accuracy': 0.0,
            'overall_accuracy': 0.0
        }

def calculate_accuracy(predicted_stocks, actual_returns):
    try:
        if not predicted_stocks or not actual_returns:
            return 0.0
            
        correct_predictions = 0
        total_predictions = len(predicted_stocks)
        
        for ticker in predicted_stocks:
            try:
                if ticker in actual_returns:
                    # Properly handle Series or scalar values
                    return_value = float(actual_returns[ticker].iloc[0] if isinstance(actual_returns[ticker], pd.Series) else actual_returns[ticker])
                    if return_value > 0:
                        correct_predictions += 1
            except Exception as e:
                print(f"Error processing accuracy for {ticker}: {str(e)}")
                continue
                
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        return accuracy
        
    except Exception as e:
        print(f"Error in calculate_accuracy: {str(e)}")
        return 0.0

def calculate_overall_accuracy(predictions, actual_returns):
    try:
        if not predictions or not actual_returns:
            return 0.0
            
        total_stocks = len(predictions['gainers']) + len(predictions['losers'])
        if total_stocks == 0:
            return 0.0
            
        correct_predictions = 0
        
        # Check gainers
        for ticker in predictions['gainers']:
            try:
                if ticker in actual_returns:
                    # Properly handle Series or scalar values
                    return_value = float(actual_returns[ticker].iloc[0] if isinstance(actual_returns[ticker], pd.Series) else actual_returns[ticker])
                    if return_value > 0:
                        correct_predictions += 1
            except Exception as e:
                print(f"Error processing gainer accuracy for {ticker}: {str(e)}")
                continue
                
        # Check losers
        for ticker in predictions['losers']:
            try:
                if ticker in actual_returns:
                    # Properly handle Series or scalar values
                    return_value = float(actual_returns[ticker].iloc[0] if isinstance(actual_returns[ticker], pd.Series) else actual_returns[ticker])
                    if return_value < 0:
                        correct_predictions += 1
            except Exception as e:
                print(f"Error processing loser accuracy for {ticker}: {str(e)}")
                continue
                
        accuracy = (correct_predictions / total_stocks) * 100 if total_stocks > 0 else 0.0
        return accuracy
        
    except Exception as e:
        print(f"Error in calculate_overall_accuracy: {str(e)}")
        return 0.0

def get_actual_returns(predictions, days=5):
    actual_returns = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for ticker in predictions['gainers'] + predictions['losers']:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                initial_price = data['Close'].iloc[0]
                final_price = data['Close'].iloc[-1]
                actual_returns[ticker] = (final_price - initial_price) / initial_price
        except Exception as e:
            print(f"Error getting returns for {ticker}: {str(e)}")
            
    return actual_returns

def load_dow_tickers():
    try:
        print("Attempting to read Excel file 'dow 30.xlsx'...")
        # Convert the Excel data to a pandas DataFrame
        df = pd.read_excel('dow 30.xlsx')
        
        print(f"Columns found in Excel file: {list(df.columns)}")
        
        # Convert column names to uppercase for case-insensitive comparison
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Look for any column containing 'TICKER'
        ticker_col = None
        for col in df.columns:
            if 'TICKER' in col:
                ticker_col = col
                break
                
        if not ticker_col:
            raise ValueError(f"Excel file must contain a column with 'Ticker' (found columns: {list(df.columns)})")
            
        # Extract and clean tickers
        tickers = df[ticker_col].dropna().str.strip().str.upper().tolist()
        
        if not tickers:
            print("No tickers found in Excel file")
            raise ValueError("No tickers found in Excel file")
            
        print(f"Successfully loaded {len(tickers)} tickers from Excel file:")
        for ticker in tickers:
            print(f"Found ticker: {ticker}")
            
        return tickers
    except Exception as e:
        print(f"Error loading Dow tickers: {str(e)}")
        print("Falling back to default tickers list...")
        # Fallback to hardcoded list if Excel read fails
        default_tickers = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 
                          'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 
                          'NVDA', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V', 'WMT']
        print(f"Using {len(default_tickers)} default tickers")
        return default_tickers

def get_stock_data(ticker):
    try:
        # Download data
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        
        if df.empty:
            print(f"No data available for {ticker}")
            return None
            
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # If we have a MultiIndex, convert to regular DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
            
        # Calculate returns
        df['Return'] = df['Close'].pct_change()
        
        try:
            # Calculate RSI manually
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD manually
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            df['MACD'] = macd - signal
            
            # Fill any NaN values in the technical indicators
            df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI value
            df['MACD'] = df['MACD'].fillna(0)  # Neutral MACD value
            
        except Exception as e:
            print(f"Error calculating technical indicators for {ticker}: {str(e)}")
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_news_sentiment(ticker):
    try:
        # Get dates for last 10 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params)
        news = response.json()

        # Enhanced sentiment scoring
        sentiment_score = 0
        for article in news:
            text = (article.get('headline', '') + " " + article.get('summary', '')).lower()
            # Positive indicators
            if any(word in text for word in ['beat', 'strong', 'positive', 'growth', 'increase', 'profit']):
                sentiment_score += 1
            # Negative indicators
            elif any(word in text for word in ['miss', 'weak', 'negative', 'decline', 'decrease', 'loss']):
                sentiment_score -= 1
        return sentiment_score
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return 0

def generate_features(ticker):
    try:
        print(f"\nGenerating features for {ticker}...")
        df = get_stock_data(ticker)
        if df is None or df.empty:
            print(f"No stock data available for {ticker}")
            return None
            
        # Get news sentiment
        try:
            sentiment = get_news_sentiment(ticker)
            print(f"News sentiment for {ticker}: {sentiment}")
        except Exception as e:
            print(f"Error getting sentiment for {ticker}, using default value: {str(e)}")
            sentiment = 0
        
        # Get the last row with complete data
        latest = df.dropna().iloc[-1]
        print(f"Latest data for {ticker}: {latest.to_dict()}")
        
        # Calculate 5-day return
        try:
            # Get scalar values from Series
            last_close = float(df['Close'].iloc[-1].item())
            five_days_ago_idx = -6 if len(df) > 5 else 0
            five_days_ago_close = float(df['Close'].iloc[five_days_ago_idx].item())
            five_day_return = (last_close - five_days_ago_close) / five_days_ago_close
            print(f"5-day return for {ticker}: {five_day_return:.4f}")
        except Exception as e:
            print(f"Error calculating 5-day return for {ticker}: {str(e)}")
            return None
        
        # Create feature dictionary with exact column names matching the GUI
        feature_dict = {
            'Ticker': ticker,
            'Score': 0.0,  # Will be calculated later
            'RSI': float(latest['RSI'].item()),
            'MACD': float(latest['MACD'].item()),
            '5d_return': five_day_return,
            'Sentiment': float(sentiment),
            'DL_Prediction': 0.0  # Will be updated later
        }
        
        # Verify all values are valid numbers
        for key, value in feature_dict.items():
            if key != 'Ticker':
                try:
                    # Convert to float to ensure it's a valid number
                    float_val = float(value)
                    if np.isnan(float_val) or np.isinf(float_val):
                        print(f"Invalid value for {ticker} - {key}: {float_val}")
                        return None
                    # Update the dictionary with the float value
                    feature_dict[key] = float_val
                    print(f"{key} for {ticker}: {float_val}")
                except Exception as e:
                    print(f"Error converting {key} to float for {ticker}: {str(e)}")
                    return None
        
        print(f"Successfully generated features for {ticker}")
        return feature_dict
        
    except Exception as e:
        print(f"Error generating features for {ticker}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    try:
        # Load Dow 30 tickers
        tickers = load_dow_tickers()
        print(f"Loaded {len(tickers)} tickers")
        
        # Generate features for each ticker
        features = []
        for ticker in tickers:
            try:
                ticker_features = generate_features(ticker)
                if ticker_features is not None:
                    features.append(ticker_features)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not features:
            print("No features generated. Exiting.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        # Get deep learning predictions
        try:
            dl_predictions = get_dl_predictions(tickers)
            # Map DL predictions to DataFrame
            df['dl_prediction'] = df['Ticker'].map(dl_predictions)
        except Exception as e:
            print(f"Error getting DL predictions: {str(e)}")
            df['dl_prediction'] = 0.0  # Default to 0 if DL predictions fail
        
        # Calculate final score with updated weights
        df['score'] = (
            0.25 * df['dl_prediction'] +  # Deep Learning Prediction
            0.20 * df['5d_return'] +      # Recent Performance
            0.20 * df['MACD'] +           # Trend
            0.20 * df['RSI'] +            # Momentum
            0.15 * df['Sentiment']        # News Sentiment
        )
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        # Print results
        print("\nTop 5 Predicted Gainers:")
        print(df[['Ticker', 'score', 'RSI', 'MACD', '5d_return', 'Sentiment', 'dl_prediction']].head().to_string())
        
        print("\nTop 5 Predicted Losers:")
        print(df[['Ticker', 'score', 'RSI', 'MACD', '5d_return', 'Sentiment', 'dl_prediction']].tail().to_string())
        
        # Track prediction accuracy
        try:
            predictions = {
                'gainers': df.head(5)['Ticker'].tolist(),
                'losers': df.tail(5)['Ticker'].tolist()
            }
            actual_returns = get_actual_returns(predictions)
            accuracy_metrics = track_prediction(
                datetime.now().strftime('%Y-%m-%d'),
                predictions,
                actual_returns
            )
            print("\nPrediction Accuracy Metrics:")
            print(f"Gainers Accuracy: {accuracy_metrics['gainers_accuracy']:.1f}%")
            print(f"Losers Accuracy: {accuracy_metrics['losers_accuracy']:.1f}%")
            print(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.1f}%")
        except Exception as e:
            print(f"Error tracking predictions: {str(e)}")
        
        return df
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()

