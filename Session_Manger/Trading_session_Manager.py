import sqlite3
import logging
from datetime import datetime, timedelta
import time
import pandas as pd
import pytz
from typing import List, Dict, Optional, Tuple
import yfinance as yf

class TradingSessionTracker:
    """
    Tracks trading sessions, market hours, and session timing for different markets
    """
    
    def __init__(self, db_path: str = "trading_data.db", log_level: str = "INFO"):
        self.db_path = db_path
        self.setup_logging(log_level)
        self.setup_database()
        
        self.collector = HistoryDataCollector(db_path=self.db_path)
        
        # Market sessions configuration (CAT - Central Africa Time, UTC+2)
        self.market_sessions = {
            'GOLD': {
                'Sydney': {'start': '00:00', 'end': '09:00'},
                'London': {'start': '10:00', 'end': '19:00'},
                'New_York': {'start': '15:00', 'end': '00:00'}
            },
            'NAS100': {
                'Pre_Market': {'start': '11:00', 'end': '16:30'},
                'Regular': {'start': '16:30', 'end': '23:00'},
                'After_Hours': {'start': '23:00', 'end': '03:00'}
            },
            'BTCUSD': {
                'Always_Open': {'start': '00:00', 'end': '23:59'}
            }
        }
        
        self.symbols = {
            'GOLD': 'GC=F',
            'NAS100': '^NDX',
            'BTCUSD': 'BTC-USD'
        }
        
        # Timezone setup for Botswana (CAT = UTC+2)
        self.local_tz = pytz.timezone('Africa/Gaborone')
        self.utc_tz = pytz.timezone('UTC')
        
        # Current active sessions
        self.active_sessions = {}
        
        # Volume tracking (in memory, no DB)
        self.volume_tracker = {}
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('session_tracker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database tables for session tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create session tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market TEXT NOT NULL,
                        session_name TEXT NOT NULL,
                        start_time INTEGER NOT NULL,
                        end_time INTEGER NOT NULL,
                        is_active BOOLEAN NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create session status log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        market TEXT NOT NULL,
                        session_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        message TEXT
                    )
                ''')
                
                # Create daily session schedule
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_schedule (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        market TEXT NOT NULL,
                        session_name TEXT NOT NULL,
                        start_time INTEGER NOT NULL,
                        end_time INTEGER NOT NULL,
                        duration_minutes INTEGER NOT NULL
                    )
                ''')
                
                # Indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_market ON session_tracking(market)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_timestamp ON session_status(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_schedule_date ON daily_schedule(date)')
                
                conn.commit()
                self.logger.info("Database setup for session tracking completed successfully")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
    
    def get_current_local_time(self) -> datetime:
        """Get current local time in Botswana (CAT)"""
        return datetime.now(self.local_tz)
    
    def get_current_utc_time(self) -> datetime:
        """Get current UTC time"""
        return datetime.now(self.utc_tz)
    
    def time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM string to minutes since midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def minutes_to_time(self, minutes: int) -> str:
        """Convert minutes since midnight to HH:MM string"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def is_session_active(self, market: str, session_name: str, current_time: datetime = None) -> bool:
        """Check if a specific session is currently active"""
        if current_time is None:
            current_time = self.get_current_local_time()
        
        if market not in self.market_sessions:
            return False
        
        if session_name not in self.market_sessions[market]:
            return False
        
        session = self.market_sessions[market][session_name]
        current_minutes = current_time.hour * 60 + current_time.minute
        start_minutes = self.time_to_minutes(session['start'])
        end_minutes = self.time_to_minutes(session['end'])
        
        # Handle sessions that cross midnight
        if start_minutes > end_minutes:
            return current_minutes >= start_minutes or current_minutes <= end_minutes
        else:
            return start_minutes <= current_minutes <= end_minutes
    
    def get_active_sessions(self, current_time: datetime = None) -> Dict[str, List[str]]:
        """Get all currently active sessions across all markets"""
        if current_time is None:
            current_time = self.get_current_local_time()
        
        active = {}
        
        for market, sessions in self.market_sessions.items():
            active_sessions = []
            for session_name in sessions:
                if self.is_session_active(market, session_name, current_time):
                    active_sessions.append(session_name)
            
            if active_sessions:
                active[market] = active_sessions
        
        return active
    
    def get_next_session(self, market: str, current_time: datetime = None) -> Optional[Tuple[str, datetime]]:
        """Get the next session for a specific market"""
        if current_time is None:
            current_time = self.get_current_local_time()
        
        if market not in self.market_sessions:
            return None
        
        sessions = self.market_sessions[market]
        current_minutes = current_time.hour * 60 + current_time.minute
        
        next_sessions = []
        
        for session_name, session in sessions.items():
            start_minutes = self.time_to_minutes(session['start'])
            
            # Calculate next occurrence
            if start_minutes > current_minutes:
                # Today
                next_time = current_time.replace(
                    hour=start_minutes // 60,
                    minute=start_minutes % 60,
                    second=0,
                    microsecond=0
                )
            else:
                # Tomorrow
                next_time = current_time.replace(
                    hour=start_minutes // 60,
                    minute=start_minutes % 60,
                    second=0,
                    microsecond=0
                ) + timedelta(days=1)
            
            next_sessions.append((session_name, next_time))
        
        # Return the earliest next session
        if next_sessions:
            return min(next_sessions, key=lambda x: x[1])
        
        return None
    
    def log_session_status(self, market: str, session_name: str, status: str, message: str = ""):
        """Log session status change"""
        try:
            timestamp = int(self.get_current_local_time().timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO session_status (timestamp, market, session_name, status, message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, market, session_name, status, message))
                conn.commit()
            
            self.logger.info(f"{market} {session_name}: {status} - {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to log session status: {e}")
    
    def update_session_tracking(self):
        """Update session tracking in database"""
        current_time = self.get_current_local_time()
        timestamp = int(current_time.timestamp())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing active sessions
                conn.execute('DELETE FROM session_tracking WHERE is_active = 1')
                
                # Add current active sessions
                for market, sessions in self.get_active_sessions(current_time).items():
                    for session_name in sessions:
                        session_info = self.market_sessions[market][session_name]
                        start_time = self.time_to_minutes(session_info['start'])
                        end_time = self.time_to_minutes(session_info['end'])
                        
                        conn.execute('''
                            INSERT INTO session_tracking 
                            (market, session_name, start_time, end_time, is_active)
                            VALUES (?, ?, ?, ?, 1)
                        ''', (market, session_name, start_time, end_time))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update session tracking: {e}")
    
    def generate_daily_schedule(self, date: str = None) -> pd.DataFrame:
        """Generate daily trading schedule"""
        if date is None:
            date = self.get_current_local_time().strftime('%Y-%m-%d')
        
        schedule = []
        
        for market, sessions in self.market_sessions.items():
            for session_name, session_info in sessions.items():
                start_minutes = self.time_to_minutes(session_info['start'])
                end_minutes = self.time_to_minutes(session_info['end'])
                
                # Calculate duration
                if start_minutes > end_minutes:
                    # Session crosses midnight
                    duration = (24 * 60 - start_minutes) + end_minutes
                else:
                    duration = end_minutes - start_minutes
                
                schedule.append({
                    'date': date,
                    'market': market,
                    'session_name': session_name,
                    'start_time': self.minutes_to_time(start_minutes),
                    'end_time': self.minutes_to_time(end_minutes),
                    'duration_minutes': duration
                })
        
        df = pd.DataFrame(schedule)
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing schedule for this date
                conn.execute('DELETE FROM daily_schedule WHERE date = ?', (date,))
                
                # Insert new schedule
                for _, row in df.iterrows():
                    # Create datetime objects in local timezone
                    local_start = self.local_tz.localize(datetime.strptime(f"{date} {row['start_time']}", '%Y-%m-%d %H:%M'))
                    local_end = self.local_tz.localize(datetime.strptime(f"{date} {row['end_time']}", '%Y-%m-%d %H:%M'))
                    
                    start_timestamp = int(local_start.timestamp())
                    end_timestamp = int(local_end.timestamp())
                    
                    conn.execute('''
                        INSERT INTO daily_schedule 
                        (date, market, session_name, start_time, end_time, duration_minutes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (date, row['market'], row['session_name'], 
                         start_timestamp, end_timestamp, row['duration_minutes']))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store daily schedule: {e}")
        
        return df.sort_values(['market', 'start_time'])
    
    def get_current_session_start(self, market: str, current_time: datetime) -> int:
        """Get the start timestamp of the earliest current active session"""
        if market not in self.market_sessions:
            return 0
        
        active = self.get_active_sessions(current_time).get(market, [])
        if not active:
            return 0
        
        min_start_minutes = min(self.time_to_minutes(self.market_sessions[market][s]['start']) for s in active)
        
        start_time = current_time.replace(
            hour=min_start_minutes // 60,
            minute=min_start_minutes % 60,
            second=0,
            microsecond=0
        )
        
        if start_time > current_time:
            start_time -= timedelta(days=1)
        
        return int(start_time.timestamp())
    
    def calculate_cum_volume(self, market: str, start_ts: int) -> int:
        """Calculate cumulative volume since start timestamp"""
        symbol = self.symbols.get(market)
        if not symbol:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT SUM(volume)
                    FROM candlestick_data
                    WHERE symbol = ? AND timeframe = '10m' AND timestamp >= ?
                '''
                result = conn.execute(query, (symbol, start_ts)).fetchone()
                return int(result[0]) if result[0] else 0
        except Exception as e:
            self.logger.error(f"Failed to calculate cum volume for {market}: {e}")
            return 0
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        current_time = self.get_current_local_time()
        active_sessions = self.get_active_sessions(current_time)
        
        summary = {
            'current_local_time': current_time.strftime('%Y-%m-%d %H:%M:%S CAT'),
            'active_sessions': active_sessions,
            'next_sessions': {},
            'session_status': {},
            'volume_summary': {}
        }
        
        # Get next sessions for each market
        for market in self.market_sessions.keys():
            next_session = self.get_next_session(market, current_time)
            if next_session:
                summary['next_sessions'][market] = {
                    'session_name': next_session[0],
                    'start_time': next_session[1].strftime('%Y-%m-%d %H:%M:%S CAT'),
                    'time_until': str(next_session[1] - current_time)
                }
        
        # Get session status for each market
        for market in self.market_sessions.keys():
            if market in active_sessions:
                summary['session_status'][market] = 'ACTIVE'
            else:
                summary['session_status'][market] = 'CLOSED'
        
        # Get volume summary for active markets
        for market in active_sessions.keys():
            if market in self.volume_tracker:
                tracker = self.volume_tracker[market]
                last_trend = tracker['period_volumes'][-1][2] if tracker['period_volumes'] else None
                summary['volume_summary'][market] = {
                    'current_cum_volume': tracker['last_cum_volume'],
                    'last_period_volume': tracker['last_period_volume'],
                    'trend': last_trend
                }
        
        return summary
    
    def monitor_sessions(self, duration_minutes: int = 60):
        """Monitor sessions for a specified duration"""
        self.logger.info(f"Starting session monitoring for {duration_minutes} minutes")
        
        start_time = self.get_current_local_time()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        previous_active = set()
        
        # Initial setup
        current_time = self.get_current_local_time()
        active_sessions = self.get_active_sessions(current_time)
        
        current_active = set()
        for market, sessions in active_sessions.items():
            for session in sessions:
                current_active.add(f"{market}_{session}")
        
        current_active_markets = set(active_sessions.keys())
        
        # Initialize volume trackers for initially active markets
        for market in current_active_markets:
            if market not in self.volume_tracker:
                symbol = self.symbols.get(market)
                if symbol:
                    session_start_ts = self.get_current_session_start(market, current_time)
                    self.collector.collect_historical_data(symbol, period='1d', interval='10m')
                    cum_vol = self.calculate_cum_volume(market, session_start_ts)
                    self.volume_tracker[market] = {
                        'session_start_ts': session_start_ts,
                        'last_cum_volume': cum_vol,
                        'last_period_volume': None,
                        'last_check_time': current_time,
                        'period_volumes': []
                    }
                    self.logger.info(f"Initialized volume tracking for {market} with cum volume {cum_vol}")
        
        previous_active_markets = current_active_markets
        
        while self.get_current_local_time() < end_time:
            try:
                current_time = self.get_current_local_time()
                active_sessions = self.get_active_sessions(current_time)
                
                # Convert to set for comparison
                current_active = set()
                for market, sessions in active_sessions.items():
                    for session in sessions:
                        current_active.add(f"{market}_{session}")
                
                current_active_markets = set(active_sessions.keys())
                
                # Detect session changes
                newly_opened = current_active - previous_active
                newly_closed = previous_active - current_active
                
                for session in newly_opened:
                    market, session_name = session.split('_', 1)
                    self.log_session_status(market, session_name, "OPENED", "Session started")
                
                for session in newly_closed:
                    market, session_name = session.split('_', 1)
                    self.log_session_status(market, session_name, "CLOSED", "Session ended")
                
                # Detect market-level changes
                newly_active_markets = current_active_markets - previous_active_markets
                newly_closed_markets = previous_active_markets - current_active_markets
                
                for market in newly_active_markets:
                    symbol = self.symbols.get(market)
                    if symbol:
                        session_start_ts = self.get_current_session_start(market, current_time)
                        self.collector.collect_historical_data(symbol, period='1d', interval='10m')
                        cum_vol = self.calculate_cum_volume(market, session_start_ts)
                        self.volume_tracker[market] = {
                            'session_start_ts': session_start_ts,
                            'last_cum_volume': cum_vol,
                            'last_period_volume': None,
                            'last_check_time': current_time,
                            'period_volumes': []
                        }
                        self.logger.info(f"{market} market opened, initial volume: {cum_vol}")
                
                # Check volume every 10 minutes for active markets
                for market in current_active_markets:
                    if market in self.volume_tracker:
                        tracker = self.volume_tracker[market]
                        if (current_time - tracker['last_check_time']) >= timedelta(minutes=10):
                            symbol = self.symbols.get(market)
                            if symbol:
                                self.collector.collect_historical_data(symbol, period='1d', interval='10m')
                            cum_vol = self.calculate_cum_volume(market, tracker['session_start_ts'])
                            period_vol = cum_vol - tracker['last_cum_volume']
                            
                            trend = None
                            if tracker['last_period_volume'] is not None:
                                if period_vol > tracker['last_period_volume']:
                                    trend = "growing"
                                elif period_vol < tracker['last_period_volume']:
                                    trend = "getting lesser"
                                else:
                                    trend = "same"
                                self.logger.info(f"{market} volume is {trend}: {period_vol} vs previous {tracker['last_period_volume']}")
                            else:
                                self.logger.info(f"{market} first period volume: {period_vol}")
                            
                            tracker['period_volumes'].append((current_time, period_vol, trend))
                            tracker['last_period_volume'] = period_vol
                            tracker['last_cum_volume'] = cum_vol
                            tracker['last_check_time'] = current_time
                
                # Handle closed markets
                for market in newly_closed_markets:
                    if market in self.volume_tracker:
                        periods = self.volume_tracker[market]['period_volumes']
                        if periods:
                            total_vol = sum(p[1] for p in periods)
                            self.logger.info(f"{market} market closed, total session volume: {total_vol}")
                        del self.volume_tracker[market]
                
                # Update tracking
                self.update_session_tracking()
                
                previous_active = current_active
                previous_active_markets = current_active_markets
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # Shorter wait on error
    
    def print_current_status(self):
        """Print current session status to console"""
        summary = self.get_session_summary()
        
        print("\n" + "="*50)
        print("TRADING SESSION STATUS")
        print("="*50)
        print(f"Current Time: {summary['current_local_time']}")
        print()
        
        print("ACTIVE SESSIONS:")
        if summary['active_sessions']:
            for market, sessions in summary['active_sessions'].items():
                for session in sessions:
                    print(f"  ✓ {market} - {session}")
        else:
            print("  No active sessions")
        print()
        
        print("NEXT SESSIONS:")
        for market, next_info in summary['next_sessions'].items():
            print(f"  {market}: {next_info['session_name']} at {next_info['start_time']}")
            print(f"    Time until: {next_info['time_until']}")
        print()
        
        print("MARKET STATUS:")
        for market, status in summary['session_status'].items():
            status_icon = "🟢" if status == "ACTIVE" else "🔴"
            print(f"  {status_icon} {market}: {status}")
        
        print()
        print("VOLUME SUMMARY:")
        if summary['volume_summary']:
            for market, vs in summary['volume_summary'].items():
                print(f"  {market}: Cum Vol {vs['current_cum_volume']}, Last 10m Vol {vs['last_period_volume']}, Trend: {vs['trend'] or 'N/A'}")
        else:
            print("  No volume data yet")
        
        print("="*50)


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
            '10m': 144,  # 24 * 6
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
    tracker = TradingSessionTracker()
    
    # Print current status
    tracker.print_current_status()
    
    # Generate today's schedule
    print("\nTODAY'S TRADING SCHEDULE:")
    print("-" * 40)
    schedule = tracker.generate_daily_schedule()
    print(schedule.to_string(index=False))
    
    # Show recent session status changes
    print("\nRECENT SESSION CHANGES:")
    print("-" * 30)
    try:
        with sqlite3.connect(tracker.db_path) as conn:
            recent_logs = pd.read_sql_query('''
                SELECT timestamp, market, session_name, status, message 
                FROM session_status 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', conn)
            
            if not recent_logs.empty:
                recent_logs['datetime'] = pd.to_datetime(recent_logs['timestamp'], unit='s')
                print(recent_logs[['datetime', 'market', 'session_name', 'status']].to_string(index=False))
            else:
                print("No recent session changes recorded")
                
    except Exception as e:
        print(f"Could not retrieve recent logs: {e}")
    
    # Monitor for a short period (demo)
    print(f"\nMonitoring sessions for 2 minutes...")
    print("(In production, you would run this continuously)")
    
    # For demo, just update once
    tracker.update_session_tracking()
    tracker.print_current_status()


if __name__ == "__main__":
    main()