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
        try:
            self.refresh_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setValue(0)
            
            # Load tickers
            self.tickers = load_dow_tickers()
            total_tickers = len(self.tickers)
            print(f"Loaded {total_tickers} tickers")
            
            # Process each ticker
            feature_list = []
            successful_tickers = 0
            failed_tickers = 0
            
            for i, ticker in enumerate(self.tickers):
                self.progress.setValue(int((i / total_tickers) * 100))
                QApplication.processEvents()  # Keep UI responsive
                print(f"\nProcessing {ticker}...")
                
                try:
                    features = generate_features(ticker)
                    if features:
                        # Verify all required fields are present
                        required_fields = ['Ticker', 'RSI', 'MACD', '5d_return', 'Sentiment']
                        if all(field in features for field in required_fields):
                            feature_list.append(features)
                            successful_tickers += 1
                            print(f"Successfully processed {ticker} with features: {features}")
                        else:
                            missing_fields = [field for field in required_fields if field not in features]
                            print(f"Missing fields for {ticker}: {missing_fields}")
                            failed_tickers += 1
                    else:
                        print(f"No features generated for {ticker}")
                        failed_tickers += 1
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    failed_tickers += 1
            
            print(f"\nProcessing complete:")
            print(f"Successful tickers: {successful_tickers}")
            print(f"Failed tickers: {failed_tickers}")
            
            if feature_list:
                print(f"\nCreating DataFrame with {len(feature_list)} stocks")
                # Create DataFrame
                df = pd.DataFrame(feature_list)
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"DataFrame shape: {df.shape}")
                
                # Calculate scores
                df['Score'] = (
                    df['5d_return'] * 0.4 +
                    df['MACD'] * 0.3 +
                    (df['RSI'] / 100) * 0.2 +
                    df['Sentiment'] * 0.1
                )
                
                print("\nScore statistics:")
                print(df[['Ticker', 'Score']].to_string())
                
                print("\nUpdating tables...")
                # Update tables
                self.update_tables(df)
                print("Tables updated")
            else:
                print("No data available for analysis")
            
            self.progress.setVisible(False)
            self.refresh_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error in refresh_data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.progress.setVisible(False)
            self.refresh_btn.setEnabled(True)
        
    def update_tables(self, df):
        try:
            print("\nSorting data...")
            print(f"Input DataFrame shape: {df.shape}")
            print(f"Input DataFrame columns: {df.columns.tolist()}")
            print(f"Input DataFrame head:\n{df.head()}")
            
            # Sort for gainers and losers
            top_gainers = df.nlargest(5, 'Score')
            top_losers = df.nsmallest(5, 'Score')
            
            print("\nTop 5 Gainers DataFrame shape:", top_gainers.shape)
            print("Top 5 Gainers:")
            print(top_gainers[['Ticker', 'Score']].to_string())
            print("\nTop 5 Losers DataFrame shape:", top_losers.shape)
            print("Top 5 Losers:")
            print(top_losers[['Ticker', 'Score']].to_string())
            
            print("\nUpdating gainers table...")
            # Update gainers table
            self.gainers_table.setRowCount(len(top_gainers))
            self.update_table_content(self.gainers_table, top_gainers)
            
            print("Updating losers table...")
            # Update losers table
            self.losers_table.setRowCount(len(top_losers))
            self.update_table_content(self.losers_table, top_losers)
            
            print("Updating full table...")
            # Update full table
            self.full_table.setRowCount(len(df))
            self.update_table_content(self.full_table, df)
            
        except Exception as e:
            print(f"Error in update_tables: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
    def update_table_content(self, table, data):
        try:
            print(f"\nUpdating table with {len(data)} rows")
            print(f"Data columns: {data.columns.tolist()}")
            
            # Reset the table first
            table.clearContents()
            table.setRowCount(len(data))
            
            # Convert DataFrame to list of lists for more reliable access
            rows = data.values.tolist()
            
            for i in range(len(rows)):
                row = rows[i]
                ticker = str(row[0])  # Ticker is first column
                score = float(row[1])  # Score is second column
                rsi = float(row[2])    # RSI is third column
                macd = float(row[3])   # MACD is fourth column
                ret = float(row[4])    # 5d_return is fifth column
                sentiment = float(row[5])  # Sentiment is sixth column
                
                print(f"Setting row {i}: {ticker} with score {score}")
                
                # Set items with proper data types
                table.setItem(i, 0, QTableWidgetItem(ticker))
                table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{rsi:.2f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{macd:.4f}"))
                table.setItem(i, 4, QTableWidgetItem(f"{ret:.4%}"))
                table.setItem(i, 5, QTableWidgetItem(f"{sentiment:.0f}"))
            
            print(f"Table row count after update: {table.rowCount()}")
            table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Error in update_table_content: {str(e)}")
            import traceback
            print(traceback.format_exc())

def main():
    app = QApplication(sys.argv)
    window = MarketMoversApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 