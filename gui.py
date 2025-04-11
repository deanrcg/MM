import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                           QTableWidgetItem, QProgressBar, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pandas as pd
from MM import load_dow_tickers, generate_features

class MarketMoversApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Movers - Dow Jones Analysis")
        self.setMinimumSize(1000, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Create tabs
        self.prediction_tab = QWidget()
        self.analysis_tab = QWidget()
        tabs.addTab(self.prediction_tab, "Predictions")
        tabs.addTab(self.analysis_tab, "Analysis")
        
        # Setup prediction tab
        self.setup_prediction_tab()
        
        # Setup analysis tab
        self.setup_analysis_tab()
        
        # Initialize data
        self.tickers = []
        self.data = pd.DataFrame()
        
    def setup_prediction_tab(self):
        layout = QVBoxLayout(self.prediction_tab)
        
        # Header
        header = QLabel("Dow Jones Market Movers")
        header.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self.refresh_data)
        btn_layout.addWidget(self.refresh_btn)
        layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Tables
        tables_layout = QHBoxLayout()
        
        # Gainers table
        gainers_layout = QVBoxLayout()
        gainers_label = QLabel("Top 5 Predicted Gainers")
        gainers_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        gainers_layout.addWidget(gainers_label)
        
        self.gainers_table = QTableWidget()
        self.gainers_table.setColumnCount(6)
        self.gainers_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment'])
        gainers_layout.addWidget(self.gainers_table)
        tables_layout.addLayout(gainers_layout)
        
        # Losers table
        losers_layout = QVBoxLayout()
        losers_label = QLabel("Top 5 Predicted Losers")
        losers_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        losers_layout.addWidget(losers_label)
        
        self.losers_table = QTableWidget()
        self.losers_table.setColumnCount(6)
        self.losers_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment'])
        losers_layout.addWidget(self.losers_table)
        tables_layout.addLayout(losers_layout)
        
        layout.addLayout(tables_layout)
        
    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)
        
        # Full stock list table
        self.full_table = QTableWidget()
        self.full_table.setColumnCount(6)
        self.full_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment'])
        layout.addWidget(self.full_table)
        
    def refresh_data(self):
        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        # Load tickers
        self.tickers = load_dow_tickers()
        total_tickers = len(self.tickers)
        
        # Process each ticker
        feature_list = []
        for i, ticker in enumerate(self.tickers):
            self.progress.setValue(int((i / total_tickers) * 100))
            features = generate_features(ticker)
            if features:
                feature_list.append(features)
        
        if feature_list:
            # Create DataFrame
            df = pd.DataFrame(feature_list)
            
            # Calculate scores
            df['Score'] = (
                df['5d_return'] * 0.4 +
                df['MACD'] * 0.3 +
                (df['RSI'] / 100) * 0.2 +
                df['Sentiment'] * 0.1
            )
            
            # Update tables
            self.update_tables(df)
        
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        
    def update_tables(self, df):
        # Sort for gainers and losers
        top_gainers = df.sort_values(by='Score', ascending=False).head(5)
        top_losers = df.sort_values(by='Score', ascending=True).head(5)
        
        # Update gainers table
        self.gainers_table.setRowCount(5)
        self.update_table_content(self.gainers_table, top_gainers)
        
        # Update losers table
        self.losers_table.setRowCount(5)
        self.update_table_content(self.losers_table, top_losers)
        
        # Update full table
        self.full_table.setRowCount(len(df))
        self.update_table_content(self.full_table, df)
        
    def update_table_content(self, table, data):
        for i, row in enumerate(data.itertuples()):
            table.setItem(i, 0, QTableWidgetItem(row.Ticker))
            table.setItem(i, 1, QTableWidgetItem(f"{row.Score:.4f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{row.RSI:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{row.MACD:.4f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{row._5d_return:.4%}"))
            table.setItem(i, 5, QTableWidgetItem(f"{row.Sentiment:.0f}"))
        
        table.resizeColumnsToContents()

def main():
    app = QApplication(sys.argv)
    window = MarketMoversApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 