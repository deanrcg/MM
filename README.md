# Market Movers Predictor

A Python application that predicts potential stock movements in the Dow Jones Industrial Average (DJIA) using technical analysis and news sentiment.

## Features

- Automatically loads Dow 30 stock tickers from Excel file
- Fetches historical price data using yfinance
- Calculates technical indicators (RSI, MACD)
- Analyzes news sentiment using Finnhub API
- Predicts top 5 potential gainers and losers

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Place your Dow 30 stock data in `dow 30.xlsx`
2. The Finnhub API key is already configured in the script

## Usage

Run the script:
```bash
python MM.py
```

## Output

The script will display:
- Top 5 predicted gainers with their scores
- Top 5 predicted losers with their scores

## Scoring System

The prediction score is calculated using:
- 40% Recent performance (5-day return)
- 30% Trend (MACD)
- 20% Momentum (RSI)
- 10% News sentiment

## Dependencies

- yfinance: Stock data
- ta: Technical analysis
- pandas: Data manipulation
- requests: API calls
- openpyxl: Excel file handling 