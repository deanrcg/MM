import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                           QTableWidgetItem, QProgressBar, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pandas as pd
from MM import load_dow_tickers, generate_features, get_actual_returns, track_prediction
from datetime import datetime

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
        
        # Accuracy metrics
        metrics_layout = QHBoxLayout()
        
        # Gainers accuracy
        gainers_metrics = QVBoxLayout()
        gainers_label = QLabel("Gainers Accuracy")
        gainers_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.gainers_accuracy = QLabel("0%")
        self.gainers_accuracy.setFont(QFont('Arial', 12))
        gainers_metrics.addWidget(gainers_label)
        gainers_metrics.addWidget(self.gainers_accuracy)
        metrics_layout.addLayout(gainers_metrics)
        
        # Losers accuracy
        losers_metrics = QVBoxLayout()
        losers_label = QLabel("Losers Accuracy")
        losers_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.losers_accuracy = QLabel("0%")
        self.losers_accuracy.setFont(QFont('Arial', 12))
        losers_metrics.addWidget(losers_label)
        losers_metrics.addWidget(self.losers_accuracy)
        metrics_layout.addLayout(losers_metrics)
        
        # Overall accuracy
        overall_metrics = QVBoxLayout()
        overall_label = QLabel("Overall Accuracy")
        overall_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.overall_accuracy = QLabel("0%")
        self.overall_accuracy.setFont(QFont('Arial', 12))
        overall_metrics.addWidget(overall_label)
        overall_metrics.addWidget(self.overall_accuracy)
        metrics_layout.addLayout(overall_metrics)
        
        layout.addLayout(metrics_layout)
        
        # Tables
        tables_layout = QHBoxLayout()
        
        # Gainers table
        gainers_layout = QVBoxLayout()
        gainers_label = QLabel("Top 5 Predicted Gainers")
        gainers_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        gainers_layout.addWidget(gainers_label)
        
        self.gainers_table = QTableWidget()
        self.gainers_table.setColumnCount(7)
        self.gainers_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment', 'DL Prediction'])
        gainers_layout.addWidget(self.gainers_table)
        tables_layout.addLayout(gainers_layout)
        
        # Losers table
        losers_layout = QVBoxLayout()
        losers_label = QLabel("Top 5 Predicted Losers")
        losers_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        losers_layout.addWidget(losers_label)
        
        self.losers_table = QTableWidget()
        self.losers_table.setColumnCount(7)
        self.losers_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment', 'DL Prediction'])
        losers_layout.addWidget(self.losers_table)
        tables_layout.addLayout(losers_layout)
        
        layout.addLayout(tables_layout)
        
    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)
        
        # Full stock list table
        self.full_table = QTableWidget()
        self.full_table.setColumnCount(7)
        self.full_table.setHorizontalHeaderLabels(['Ticker', 'Score', 'RSI', 'MACD', '5d Return', 'Sentiment', 'DL Prediction'])
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
            print("\nUpdating tables with DataFrame:")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            
            # Sort by score and get top 5 gainers and losers
            gainers = df.nlargest(5, 'Score')
            losers = df.nsmallest(5, 'Score')
            
            print(f"\nTop 5 gainers: {gainers['Ticker'].tolist()}")
            print(f"Top 5 losers: {losers['Ticker'].tolist()}")
            
            # Update tables
            self.update_table_content(self.gainers_table, gainers)
            self.update_table_content(self.losers_table, losers)
            self.update_table_content(self.full_table, df)
            
            # Calculate accuracy metrics
            try:
                # Get predictions as a list of tuples (ticker, prediction)
                predictions = {
                    'gainers': gainers['Ticker'].tolist(),
                    'losers': losers['Ticker'].tolist()
                }
                
                # Get actual returns for these tickers
                actual_returns = get_actual_returns(predictions)
                
                # Track predictions and get accuracy metrics
                accuracy_metrics = track_prediction(
                    datetime.now().strftime('%Y-%m-%d'),
                    predictions,
                    actual_returns
                )
                
                # Update accuracy labels
                self.gainers_accuracy.setText(f"{accuracy_metrics['gainers_accuracy']:.1f}%")
                self.losers_accuracy.setText(f"{accuracy_metrics['losers_accuracy']:.1f}%")
                self.overall_accuracy.setText(f"{accuracy_metrics['overall_accuracy']:.1f}%")
                
            except Exception as e:
                print(f"Error calculating accuracy metrics: {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Set default values if calculation fails
                self.gainers_accuracy.setText("N/A")
                self.losers_accuracy.setText("N/A")
                self.overall_accuracy.setText("N/A")
            
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
            
            # Map column names to indices
            column_map = {
                'Ticker': 0,
                'Score': 1,
                'RSI': 2,
                'MACD': 3,
                '5d_return': 4,
                'Sentiment': 5,
                'DL_Prediction': 6
            }
            
            # Ensure we have the correct number of rows
            if table.rowCount() != len(data):
                table.setRowCount(len(data))
            
            # Convert DataFrame to list of dictionaries for easier access
            rows = data.to_dict('records')
            
            for i, row in enumerate(rows):
                try:
                    # Set items with proper data types and formatting
                    table.setItem(i, column_map['Ticker'], QTableWidgetItem(str(row['Ticker'])))
                    table.setItem(i, column_map['Score'], QTableWidgetItem(f"{float(row['Score']):.4f}"))
                    table.setItem(i, column_map['RSI'], QTableWidgetItem(f"{float(row['RSI']):.2f}"))
                    table.setItem(i, column_map['MACD'], QTableWidgetItem(f"{float(row['MACD']):.4f}"))
                    table.setItem(i, column_map['5d_return'], QTableWidgetItem(f"{float(row['5d_return']):.2%}"))
                    table.setItem(i, column_map['Sentiment'], QTableWidgetItem(f"{float(row['Sentiment']):.0f}"))
                    table.setItem(i, column_map['DL_Prediction'], QTableWidgetItem(f"{float(row['DL_Prediction']):.2%}"))
                    
                    print(f"Set row {i} with ticker {row['Ticker']}")
                except Exception as e:
                    print(f"Error setting row {i}: {str(e)}")
                    continue
            
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