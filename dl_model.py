import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class StockPredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # LSTM layers for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        return output

def prepare_data(ticker, lookback=30):
    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback*2)  # Extra data for calculating returns
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        return None, None
    
    # Calculate features
    df['Return'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd - signal
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['MA20'] + (df['20dSTD'] * 2)
    df['LowerBand'] = df['MA20'] - (df['20dSTD'] * 2)
    
    # Normalize features
    features = df[['Close', 'Volume', 'RSI', 'MACD', 'Volume_Change', 'UpperBand', 'LowerBand']].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(df['Return'].iloc[i])  # Predict next day's return
    
    return np.array(X), np.array(y)

def train_model(ticker, epochs=50, batch_size=32):
    # Prepare data
    X, y = prepare_data(ticker)
    if X is None:
        return None
    
    # Create dataset and dataloader
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = StockPredictionModel(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model

def predict_future(model, ticker, days=5):
    # Get latest data
    X, _ = prepare_data(ticker)
    if X is None:
        return None
    
    # Get the most recent sequence
    latest_sequence = X[-1:]
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(torch.FloatTensor(latest_sequence))
    
    return prediction.item()

def get_dl_predictions(tickers):
    predictions = {}
    for ticker in tickers:
        print(f"\nTraining model for {ticker}...")
        model = train_model(ticker)
        if model:
            prediction = predict_future(model, ticker)
            predictions[ticker] = prediction
        else:
            predictions[ticker] = 0.0
    
    return predictions 