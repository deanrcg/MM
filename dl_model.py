import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import ta
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

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

def prepare_data(ticker, lookback=60):
    try:
        # Fetch historical data
        df = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if df.empty:
            print(f"No data available for {ticker}")
            return None
            
        # Calculate technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_bands()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Select features
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'Returns']
        df = df[features]
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:(i + lookback)])
            y.append(scaled_data[i + lookback, -1])  # Predict next day's return
            
        return np.array(X), np.array(y), scaler
        
    except Exception as e:
        print(f"Error preparing data for {ticker}: {str(e)}")
        return None

def train_model(X, y, epochs=100, batch_size=32):
    try:
        # Split data into training and validation sets
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model and optimizer
        model = StockPredictionModel(input_size=X.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        return model
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def predict_future(model, X, scaler, days=5):
    try:
        model.eval()
        predictions = []
        current_sequence = X[-1:]  # Start with the last sequence
        
        for _ in range(days):
            with torch.no_grad():
                prediction = model(torch.FloatTensor(current_sequence))
                predictions.append(prediction.item())
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, -1] = prediction.item()  # Update returns
            
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), scaler.n_features_in_))
        dummy_array[:, -1] = predictions
        predictions = scaler.inverse_transform(dummy_array)[:, -1]
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

def process_ticker(ticker):
    try:
        print(f"\nProcessing {ticker}...")
        # Prepare data
        X, y, scaler = prepare_data(ticker)
        if X is None:
            return ticker, 0.0
            
        # Train model
        model = train_model(X, y)
        if model is None:
            return ticker, 0.0
            
        # Make prediction
        prediction = predict_future(model, X, scaler)
        if prediction is None:
            return ticker, 0.0
            
        return ticker, prediction[-1]  # Return the last day's prediction
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return ticker, 0.0

def get_dl_predictions(tickers, num_workers=4):
    """
    Get deep learning predictions for multiple tickers using parallel processing.
    
    Args:
        tickers (list): List of stock tickers
        num_workers (int): Number of parallel workers
        
    Returns:
        dict: Dictionary of ticker: prediction pairs
    """
    predictions = {}
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Process tickers in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_ticker, tickers))
    
    # Collect results
    for ticker, prediction in results:
        predictions[ticker] = prediction
        
    return predictions 