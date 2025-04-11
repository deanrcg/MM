# Technology Stack & AI Components

## Core Technologies

### Programming Languages
- Python 3.x (Primary language)
- JavaScript/Node.js (Package management support)

### Data Processing & Analysis
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- yfinance: Yahoo Finance API wrapper for stock data
- ta: Technical analysis indicators

### API Integration
- Finnhub API: Real-time stock market data and news
- requests: HTTP library for API calls

### File Handling
- openpyxl: Excel file operations
- python-dotenv: Environment variable management

## AI & Machine Learning Components

### Technical Analysis
- RSI (Relative Strength Index): Momentum indicator
- MACD (Moving Average Convergence Divergence): Trend indicator
- Custom scoring algorithm combining multiple indicators

### Natural Language Processing
- Custom sentiment analysis on news headlines and summaries
- Keyword-based scoring system for news sentiment

## Development Tools

### Version Control
- Git
- GitHub

### Package Management
- pip (Python packages)
- npm (Node.js packages)

### Documentation
- Markdown
- JSON

## Data Sources

### Market Data
- Yahoo Finance (via yfinance)
  - Historical price data
  - Trading volumes
  - Market indicators

### News & Sentiment
- Finnhub API
  - Company news
  - Market news
  - Press releases

## Scoring System

### Technical Components (90%)
- Recent Performance (40%): 5-day returns
- Trend Analysis (30%): MACD
- Momentum (20%): RSI

### AI Components (10%)
- News Sentiment Analysis
  - Positive keywords: beat, strong, positive, growth, increase, profit
  - Negative keywords: miss, weak, negative, decline, decrease, loss

## Future Enhancement Opportunities

### AI/ML Improvements
- Implement machine learning models for price prediction
- Add natural language processing for more sophisticated sentiment analysis
- Integrate deep learning for pattern recognition

### Technical Additions
- Add more technical indicators
- Implement backtesting capabilities
- Add real-time market data streaming 