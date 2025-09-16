import yfinance as yf
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os

class HistoryDataCollector:
    """
    Collects and stores historical candlestick data for trading analysis
    """
    
    def __init__(self, db_path: str = "trading_data.db", log_level: str = "INFO"):
        self.db_path = db_path
        self.setup_logging(log_level)
        self.setup_database()
        
        # Default configuration
        self.config = {
            'symbols': {
                'commodity': ['GC=F'],
                'index': ['^NDX'],
                'crypto': ['BTC-USD']
            },
            'symbol_names': {
                'GC=F': 'GOLD',
                '^NDX': 'NAS100',
                'BTC-USD': 'BTCUSD'
            },
            'timeframes': ['1h', '1d'],
            'default_period': '60d',
            'batch_size': 1000,
            'retry_attempts': 3,
            'retry_delay': 5
        }
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create candlestick data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS candlestick_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                # Create data quality table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_quality (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        date DATE NOT NULL,
                        missing_count INTEGER DEFAULT 0,
                        total_expected INTEGER DEFAULT 0,
                        quality_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON candlestick_data(symbol, timeframe)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON candlestick_data(timestamp)')
                
                conn.commit()
                self.logger.info("Database setup completed successfully")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
    
    def collect_historical_data(self, symbol: str, period: str = '60d', 
                               interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Collect historical data for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD=X')
            period: Period to fetch (e.g., '60d', '1y')
            interval: Data interval (e.g., '1h', '1d')
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        for attempt in range(self.config['retry_attempts']):
            try:
                self.logger.info(f"Fetching {symbol} data - Attempt {attempt + 1}")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    self.logger.warning(f"No data received for {symbol}")
                    return None
                
                # Clean and prepare data
                data = self.clean_data(data, symbol)
                
                # Store in database
                self.store_data(data, symbol, interval)
                
                self.logger.info(f"Successfully collected {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(self.config['retry_delay'])
                else:
                    self.logger.error(f"All attempts failed for {symbol}")
                    return None
    
    def clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate the collected data
        
        Args:
            data: Raw data from API
            symbol: Symbol name for logging
            
        Returns:
            Cleaned DataFrame
        """
        original_len = len(data)
        
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Validate OHLC logic (High >= Low, etc.)
        if all(col in data.columns for col in price_cols):
            valid_mask = (
                (data['High'] >= data['Low']) &
                (data['High'] >= data['Open']) &
                (data['High'] >= data['Close']) &
                (data['Low'] <= data['Open']) &
                (data['Low'] <= data['Close'])
            )
            data = data[valid_mask]
        
        cleaned_len = len(data)
        if cleaned_len < original_len:
            self.logger.warning(f"Removed {original_len - cleaned_len} invalid records for {symbol}")
        
        return data
    
    def store_data(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """
        Store cleaned data in database
        
        Args:
            data: Cleaned DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                records_stored = 0
                
                for timestamp, row in data.iterrows():
                    try:
                        # Convert timestamp to Unix timestamp
                        unix_timestamp = int(timestamp.timestamp())
                        
                        # Prepare data tuple
                        record = (
                            symbol,
                            timeframe,
                            unix_timestamp,
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row.get('Volume', 0))
                        )
                        
                        # Insert or ignore if duplicate
                        conn.execute('''
                            INSERT OR IGNORE INTO candlestick_data 
                            (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', record)
                        
                        records_stored += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error storing individual record: {e}")
                        continue
                
                conn.commit()
                self.logger.info(f"Stored {records_stored} records for {symbol} {timeframe}")
                
        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            raise
    
    def collect_all_symbols(self, period: str = '60d') -> Dict[str, bool]:
        """
        Collect data for all configured symbols
        
        Args:
            period: Period to collect data for
            
        Returns:
            Dictionary with success status for each symbol
        """
        results = {}
        
        self.logger.info(f"Starting collection for {sum(len(lst) for lst in self.config['symbols'].values())} symbols")
        
        for asset_type, sym_list in self.config['symbols'].items():
            for symbol in sym_list:
                for timeframe in self.config['timeframes']:
                    try:
                        data = self.collect_historical_data(symbol, period, timeframe)
                        results[f"{symbol}_{timeframe}"] = data is not None
                        
                        # Small delay between requests to be respectful to API
                        time.sleep(1)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to collect {symbol} {timeframe}: {e}")
                        results[f"{symbol}_{timeframe}"] = False
        
        # Log summary
        successful = sum(results.values())
        total = len(results)
        self.logger.info(f"Collection completed: {successful}/{total} successful")
        
        return results
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve latest data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of latest records to retrieve
            
        Returns:
            DataFrame with latest data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM candlestick_data
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                
                data = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                
                if not data.empty:
                    # Convert timestamp back to datetime
                    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                    data.set_index('datetime', inplace=True)
                    data.drop('timestamp', axis=1, inplace=True)
                    
                    # Sort chronologically
                    data = data.sort_index()
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return pd.DataFrame()
    
    def update_data_quality(self, symbol: str, timeframe: str):
        """
        Calculate and store data quality metrics
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get data count for today
                query = '''
                    SELECT COUNT(*) as count
                    FROM candlestick_data
                    WHERE symbol = ? AND timeframe = ?
                    AND date(timestamp, 'unixepoch') = date('now')
                '''
                
                result = conn.execute(query, (symbol, timeframe)).fetchone()
                actual_count = result[0] if result else 0
                
                # Calculate expected count based on timeframe
                expected_count = self.get_expected_count(timeframe)
                
                # Calculate quality score
                quality_score = min(actual_count / expected_count, 1.0) if expected_count > 0 else 0.0
                
                # Store quality metrics
                conn.execute('''
                    INSERT OR REPLACE INTO data_quality
                    (symbol, timeframe, date, missing_count, total_expected, quality_score)
                    VALUES (?, ?, date('now'), ?, ?, ?)
                ''', (symbol, timeframe, expected_count - actual_count, expected_count, quality_score))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating data quality: {e}")
    
    def get_expected_count(self, timeframe: str) -> int:
        """
        Calculate expected number of data points per day for a timeframe
        
        Args:
            timeframe: Data timeframe
            
        Returns:
            Expected count of data points
        """
        timeframe_map = {
            '1m': 1440,  # 24 * 60
            '5m': 288,   # 24 * 12
            '15m': 96,   # 24 * 4
            '30m': 48,   # 24 * 2
            '1h': 24,    # 24
            '4h': 6,     # 6
            '1d': 1      # 1
        }
        
        return timeframe_map.get(timeframe, 24)
    
    def run_daily_collection(self):
        """
        Main method to run daily data collection
        """
        self.logger.info("Starting daily data collection")
        
        try:
            # Collect data for all symbols
            results = self.collect_all_symbols()
            
            # Update data quality metrics
            for sym_list in self.config['symbols'].values():
                for symbol in sym_list:
                    for timeframe in self.config['timeframes']:
                        self.update_data_quality(symbol, timeframe)
            
            # Log final summary
            successful = sum(results.values())
            total = len(results)
            self.logger.info(f"Daily collection completed: {successful}/{total} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Daily collection failed: {e}")
            raise


def main():
    """
    Main execution function
    """
    collector = HistoryDataCollector()
    
    # Run daily collection
    results = collector.run_daily_collection()
    
    # Print results by asset type
    print("\n=== Data Collection Results ===")
    
    # Group results by asset type
    for asset_type, symbols in collector.config['symbols'].items():
        print(f"\n{asset_type.upper()}:")
        for symbol in symbols:
            symbol_name = collector.config['symbol_names'].get(symbol, symbol)
            for tf in collector.config['timeframes']:
                key = f"{symbol}_{tf}"
                if key in results:
                    status = "✓" if results[key] else "✗"
                    print(f"  {status} {symbol_name} ({tf})")
    
    # Show sample data for different asset types
    print("\n=== Sample Data ===")
    
    # Commodity sample
    print("\n--- GOLD (1h) ---")
    sample_gold = collector.get_latest_data('GC=F', '1h', 3)
    if not sample_gold.empty:
        print(sample_gold)
    
    # Index sample
    print("\n--- NAS100 (1d) ---")
    sample_nas = collector.get_latest_data('^NDX', '1d', 3)
    if not sample_nas.empty:
        print(sample_nas)
    
    # Crypto sample
    print("\n--- BTCUSD (1d) ---")
    sample_btc = collector.get_latest_data('BTC-USD', '1d', 3)
    if not sample_btc.empty:
        print(sample_btc)


if __name__ == "__main__":
    main()