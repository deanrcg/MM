# Market Movers - Technical Stack

## Overview
Market Movers is a desktop application built with PyQt6 that analyzes Dow Jones stocks to predict top gainers and losers. The application combines technical analysis, news sentiment analysis, and deep learning predictions to provide comprehensive market insights.

## Core Technologies

### Frontend
- **PyQt6**: Modern GUI framework for desktop applications
- **QTableWidget**: For displaying stock data in tabular format
- **QProgressBar**: For showing data loading progress
- **QTabWidget**: For organizing different views (Predictions and Analysis)

### Backend
- **Python 3.13**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **yfinance**: Stock data retrieval
- **ta**: Technical analysis indicators
- **PyTorch**: Deep learning model implementation
- **scikit-learn**: Data preprocessing and normalization

## Key Features

### Data Collection
- Real-time stock data from Yahoo Finance
- Historical price data for technical analysis
- News sentiment analysis from Finnhub API

### Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 5-day returns calculation
- Bollinger Bands (in deep learning model)

### Deep Learning Integration
- LSTM-based architecture with attention mechanism
- Multi-feature input (technical indicators + sentiment)
- Adaptive learning from historical performance
- Real-time prediction updates

### Prediction System
- Weighted scoring algorithm:
  - 25% Deep Learning Prediction
  - 20% Recent Performance (5-day return)
  - 20% Trend (MACD)
  - 20% Momentum (RSI)
  - 15% News Sentiment

### Accuracy Tracking
- Real-time accuracy metrics
- Historical performance tracking
- Separate metrics for gainers and losers
- Overall prediction accuracy

## Recent Improvements
1. Fixed table display issues
2. Added proper MultiIndex DataFrame handling
3. Improved error handling and data validation
4. Enhanced deep learning integration
5. Added comprehensive accuracy tracking

## Future Enhancements
1. Additional technical indicators
2. Advanced visualization features
3. Real-time data updates
4. Machine learning model optimization
5. User customization options

## Dependencies
```
yfinance>=0.2.36
pandas>=2.0.0
ta>=0.10.2
requests>=2.31.0
python-dotenv>=1.0.0
openpyxl>=3.1.2
PyQt6>=6.4.0
torch>=2.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

## Project Structure
```
MM/
├── MM.py              # Main application logic
├── gui.py             # GUI implementation
├── dl_model.py        # Deep learning model
├── requirements.txt   # Dependencies
└── TECH_STACK.md      # Technical documentation
``` 