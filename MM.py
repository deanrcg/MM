import yfinance as yf
import ta
import requests
import pandas as pd
from datetime import datetime, timedelta
import openpyxl
import numpy as np

# Finnhub API Configuration
FINNHUB_API_KEY = "cvsepvpr01qhup0qc89gcvsepvpr01qhup0qc8a0"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

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
    df = get_stock_data(ticker)
    if df is None or df.empty:
        return None
        
    try:
        sentiment = get_news_sentiment(ticker)
        
        # Get the last row with complete data
        latest = df.dropna().iloc[-1]
        
        # Calculate 5-day return
        last_close = df['Close'].iloc[-1].iloc[0] if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
        five_days_ago_idx = -6 if len(df) > 5 else 0
        five_days_ago_close = df['Close'].iloc[five_days_ago_idx].iloc[0] if isinstance(df['Close'].iloc[five_days_ago_idx], pd.Series) else df['Close'].iloc[five_days_ago_idx]
        five_day_return = (last_close - five_days_ago_close) / five_days_ago_close
        
        # Ensure all values are scalar using iloc[0] for Series values
        feature_dict = {
            'Ticker': ticker,
            'RSI': latest['RSI'].iloc[0] if isinstance(latest['RSI'], pd.Series) else latest['RSI'],
            'MACD': latest['MACD'].iloc[0] if isinstance(latest['MACD'], pd.Series) else latest['MACD'],
            '5d_return': five_day_return,
            'Sentiment': float(sentiment)
        }
        
        # Verify all values are valid numbers using numpy
        for key, value in feature_dict.items():
            if key != 'Ticker' and (np.isnan(value) or np.isinf(value)):
                print(f"Invalid value for {ticker} - {key}: {value}")
                return None
        
        return feature_dict
    except Exception as e:
        print(f"Error generating features for {ticker}: {str(e)}")
        return None

def main():
    # Load Dow 30 tickers
    dow_tickers = load_dow_tickers()
    print(f"Analyzing {len(dow_tickers)} Dow Jones stocks...")
    
    # Generate features for all tickers
    feature_list = []
    for ticker in dow_tickers:
        print(f"\nProcessing {ticker}...")
        features = generate_features(ticker)
        if features:
            feature_list.append(features)
            print(f"Successfully processed {ticker}")
    
    if not feature_list:
        print("\nNo data available for analysis")
        return
        
    df = pd.DataFrame(feature_list)
    
    # Calculate score with adjusted weights
    df['Score'] = (
        df['5d_return'] * 0.4 +  # Recent performance (40%)
        df['MACD'] * 0.3 +       # Trend (30%)
        (df['RSI'] / 100) * 0.2 +  # Momentum (20%) - Normalize RSI to 0-1 range
        df['Sentiment'] * 0.1    # News sentiment (10%)
    )
    
    # Get top predictions
    top_5_up = df.sort_values(by='Score', ascending=False).head(5)
    top_5_down = df.sort_values(by='Score', ascending=True).head(5)
    
    print("\n📈 Top 5 Predicted Gainers:")
    print(top_5_up[['Ticker', 'Score', 'RSI', 'MACD', '5d_return', 'Sentiment']].round(4).to_string())
    
    print("\n📉 Top 5 Predicted Losers:")
    print(top_5_down[['Ticker', 'Score', 'RSI', 'MACD', '5d_return', 'Sentiment']].round(4).to_string())

if __name__ == "__main__":
    main()

