import sqlite3
import pandas as pd
import numpy as np
import logging
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass
from enum import Enum

class PositionSizingMethod(Enum):
    MICRO_CONSERVATIVE = "micro_conservative"      # For $50-200 accounts
    MICRO_AGGRESSIVE = "micro_aggressive"          # For $200-500 accounts  
    SMALL_STEADY = "small_steady"                  # For $500-2000 accounts
    PERCENTAGE_RISK = "percentage_risk"            # Traditional % risk
    ATR_VOLATILITY = "atr_volatility"             # Volatility based
    KELLY_CRITERION = "kelly_criterion"            # Mathematical optimal

class AccountTier(Enum):
    MICRO = "micro"          # $50 - $500
    SMALL = "small"          # $500 - $2,000  
    MEDIUM = "medium"        # $2,000 - $10,000
    LARGE = "large"          # $10,000+

@dataclass
class Position:
    symbol: str
    entry_price: float
    lot_size: float           # Changed from position_size to lot_size for forex
    stop_loss: float
    take_profit: float
    direction: str
    entry_date: datetime
    risk_amount: float
    expected_profit: float
    status: str = 'open'
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    actual_profit: Optional[float] = None

@dataclass
class RiskParameters:
    account_balance: float
    risk_per_trade_pct: float
    max_portfolio_risk_pct: float
    max_positions: int
    min_risk_reward: float
    base_lot_size: float      # Base lot size for the account tier
    profit_target: float      # Target profit per trade

class RiskManager:
    """
    Realistic Risk Management System for Small Trading Accounts
    """
    
    def __init__(self, db_path: str = "trading_data.db", log_level: str = "INFO"):
        self.db_path = db_path
        self.setup_logging(log_level)
        self.setup_database()
        self.positions: List[Position] = []
        
        # Realistic risk profiles for small accounts
        self.risk_profiles = {
            AccountTier.MICRO: RiskParameters(
                account_balance=100,       # Default $100 account
                risk_per_trade_pct=0.05,   # 5% ($5 risk on $100)
                max_portfolio_risk_pct=0.15, # 15% max exposure
                max_positions=2,           # Max 2 positions
                min_risk_reward=1.0,       # 1:1 minimum R:R
                base_lot_size=0.01,        # 0.01 lots
                profit_target=2.0          # $2 profit target
            ),
            AccountTier.SMALL: RiskParameters(
                account_balance=1000,      # Default $1,000 account
                risk_per_trade_pct=0.03,   # 3% ($30 risk on $1,000)
                max_portfolio_risk_pct=0.12, # 12% max exposure
                max_positions=3,           # Max 3 positions
                min_risk_reward=1.5,       # 1.5:1 minimum R:R
                base_lot_size=0.05,        # 0.05 lots
                profit_target=15.0         # $15 profit target
            ),
            AccountTier.MEDIUM: RiskParameters(
                account_balance=5000,      # Default $5,000 account
                risk_per_trade_pct=0.02,   # 2% ($100 risk on $5,000)
                max_portfolio_risk_pct=0.10, # 10% max exposure
                max_positions=5,           # Max 5 positions
                min_risk_reward=2.0,       # 2:1 minimum R:R
                base_lot_size=0.10,        # 0.10 lots
                profit_target=50.0         # $50 profit target
            ),
            AccountTier.LARGE: RiskParameters(
                account_balance=25000,     # Default $25,000 account
                risk_per_trade_pct=0.015,  # 1.5% ($375 risk on $25,000)
                max_portfolio_risk_pct=0.08, # 8% max exposure
                max_positions=8,           # Max 8 positions
                min_risk_reward=2.5,       # 2.5:1 minimum R:R
                base_lot_size=0.25,        # 0.25 lots
                profit_target=150.0        # $150 profit target
            )
        }
        
        self.current_profile = self.risk_profiles[AccountTier.MICRO]
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('risk_management.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize database tables for risk management"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Account progression tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS account_progression (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        account_balance REAL NOT NULL,
                        account_tier TEXT NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        total_profit REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Risk settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        account_tier TEXT NOT NULL,
                        account_balance REAL NOT NULL,
                        risk_per_trade_pct REAL NOT NULL,
                        max_portfolio_risk_pct REAL NOT NULL,
                        max_positions INTEGER NOT NULL,
                        base_lot_size REAL NOT NULL,
                        profit_target REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Positions table (updated for forex lots)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        lot_size REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        direction TEXT NOT NULL,
                        entry_date TEXT NOT NULL,
                        risk_amount REAL NOT NULL,
                        expected_profit REAL NOT NULL,
                        status TEXT DEFAULT 'open',
                        exit_price REAL,
                        exit_date TEXT,
                        actual_profit REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Position sizing calculations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_sizing_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        method TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        calculated_lot_size REAL NOT NULL,
                        risk_amount REAL NOT NULL,
                        expected_profit REAL NOT NULL,
                        account_balance REAL NOT NULL,
                        calculation_date TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Growth milestones table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS growth_milestones (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        milestone_balance REAL NOT NULL,
                        achieved_date TEXT,
                        days_to_achieve INTEGER,
                        trades_to_achieve INTEGER,
                        tier_upgrade TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_account_date ON account_progression(date)')
                
                conn.commit()
                self.logger.info("Risk management database setup completed")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
    
    def set_account_profile(self, account_balance: float):
        """Set risk parameters based on realistic account balance"""
        self.current_profile.account_balance = account_balance
        
        if account_balance <= 500:
            self.current_profile = copy.copy(self.risk_profiles[AccountTier.MICRO])
            self.current_profile.account_balance = account_balance
            tier = "MICRO"
        elif account_balance <= 2000:
            self.current_profile = copy.copy(self.risk_profiles[AccountTier.SMALL])
            self.current_profile.account_balance = account_balance
            tier = "SMALL"
        elif account_balance <= 10000:
            self.current_profile = copy.copy(self.risk_profiles[AccountTier.MEDIUM])
            self.current_profile.account_balance = account_balance
            tier = "MEDIUM"
        else:
            self.current_profile = copy.copy(self.risk_profiles[AccountTier.LARGE])
            self.current_profile.account_balance = account_balance
            tier = "LARGE"
        
        # Adjust lot size based on exact balance for micro accounts
        if account_balance <= 500:
            if account_balance < 100:
                self.current_profile.base_lot_size = 0.01
                self.current_profile.profit_target = max(0.5, account_balance * 0.01)
            elif account_balance < 200:
                self.current_profile.base_lot_size = 0.01
                self.current_profile.profit_target = max(1.0, account_balance * 0.015)
            elif account_balance < 500:
                self.current_profile.base_lot_size = 0.02
                self.current_profile.profit_target = max(2.0, account_balance * 0.02)
        
        self.logger.info(f"Set {tier} account profile for ${account_balance:.2f}")
        self.save_risk_settings()
    
    def save_risk_settings(self):
        """Save current risk settings to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Determine tier name
                tier_name = "MICRO" if self.current_profile.account_balance <= 500 else \
                           "SMALL" if self.current_profile.account_balance <= 2000 else \
                           "MEDIUM" if self.current_profile.account_balance <= 10000 else "LARGE"
                
                conn.execute('''
                    INSERT INTO risk_settings 
                    (account_tier, account_balance, risk_per_trade_pct, max_portfolio_risk_pct, 
                     max_positions, base_lot_size, profit_target)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tier_name, self.current_profile.account_balance, self.current_profile.risk_per_trade_pct,
                    self.current_profile.max_portfolio_risk_pct, self.current_profile.max_positions,
                    self.current_profile.base_lot_size, self.current_profile.profit_target
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save risk settings: {e}")
    
    def calculate_micro_conservative_sizing(self, symbol: str, entry_price: float, stop_loss: float) -> Tuple[float, float, float]:
        """
        Micro account conservative sizing - prioritizes capital preservation
        For $50-$200 accounts with 0.01 lot sizes and $0.50-$2 profit targets
        """
        # Fixed small lot size for micro accounts
        lot_size = 0.01
        
        # Calculate pip value (assuming forex pairs)
        pip_value = self.get_pip_value(symbol, lot_size)
        
        # Calculate stop loss in pips
        stop_pips = abs(entry_price - stop_loss) * 10000  # For 4-digit quotes
        if 'JPY' in symbol:
            stop_pips = abs(entry_price - stop_loss) * 100  # For JPY pairs
        
        # Risk amount (conservative)
        risk_amount = stop_pips * pip_value
        
        # Conservative profit target (1:1 or 1:1.5 R:R)
        profit_pips = stop_pips * 1.2  # 1.2:1 R:R
        expected_profit = profit_pips * pip_value
        
        # Cap risk at 5% of micro account
        max_risk = self.current_profile.account_balance * 0.05
        if risk_amount > max_risk:
            # Reduce lot size if risk is too high
            lot_size = 0.01 * (max_risk / risk_amount)
            lot_size = max(0.01, round(lot_size, 2))
            risk_amount = stop_pips * self.get_pip_value(symbol, lot_size)
            expected_profit = profit_pips * self.get_pip_value(symbol, lot_size)
        
        return lot_size, risk_amount, expected_profit
    
    def calculate_micro_aggressive_sizing(self, symbol: str, entry_price: float, stop_loss: float) -> Tuple[float, float, float]:
        """
        Micro account aggressive sizing - faster growth but higher risk
        For $200-$500 accounts with 0.01-0.03 lot sizes
        """
        # Slightly larger lot size for aggressive micro accounts
        base_lot = 0.02 if self.current_profile.account_balance >= 300 else 0.01
        
        pip_value = self.get_pip_value(symbol, base_lot)
        
        # Calculate stop loss in pips
        stop_pips = abs(entry_price - stop_loss) * 10000
        if 'JPY' in symbol:
            stop_pips = abs(entry_price - stop_loss) * 100
        
        # Risk amount (more aggressive)
        risk_amount = stop_pips * pip_value
        
        # Aggressive profit target (1:2 R:R)
        profit_pips = stop_pips * 2.0
        expected_profit = profit_pips * pip_value
        
        # Cap risk at 8% of account for aggressive approach
        max_risk = self.current_profile.account_balance * 0.08
        if risk_amount > max_risk:
            lot_size = base_lot * (max_risk / risk_amount)
            lot_size = max(0.01, round(lot_size, 2))
            risk_amount = stop_pips * self.get_pip_value(symbol, lot_size)
            expected_profit = profit_pips * self.get_pip_value(symbol, lot_size)
        else:
            lot_size = base_lot
        
        return lot_size, risk_amount, expected_profit
    
    def calculate_small_steady_sizing(self, symbol: str, entry_price: float, stop_loss: float) -> Tuple[float, float, float]:
        """
        Small account steady growth sizing - balanced approach
        For $500-$2000 accounts with 0.02-0.10 lot sizes
        """
        # Progressive lot sizing based on account balance
        if self.current_profile.account_balance < 800:
            base_lot = 0.02
        elif self.current_profile.account_balance < 1200:
            base_lot = 0.03
        elif self.current_profile.account_balance < 1600:
            base_lot = 0.05
        else:
            base_lot = 0.07
        
        pip_value = self.get_pip_value(symbol, base_lot)
        
        stop_pips = abs(entry_price - stop_loss) * 10000
        if 'JPY' in symbol:
            stop_pips = abs(entry_price - stop_loss) * 100
        
        risk_amount = stop_pips * pip_value
        
        # Target 1:2.5 R:R for steady growth
        profit_pips = stop_pips * 2.5
        expected_profit = profit_pips * pip_value
        
        # Cap risk at 3% for steady approach
        max_risk = self.current_profile.account_balance * 0.03
        if risk_amount > max_risk:
            lot_size = base_lot * (max_risk / risk_amount)
            lot_size = max(0.01, round(lot_size, 2))
            risk_amount = stop_pips * self.get_pip_value(symbol, lot_size)
            expected_profit = profit_pips * self.get_pip_value(symbol, lot_size)
        else:
            lot_size = base_lot
        
        return lot_size, risk_amount, expected_profit
    
    def calculate_percentage_risk_sizing(self, symbol: str, entry_price: float, stop_loss: float) -> Tuple[float, float, float]:
        """Traditional percentage risk position sizing"""
        risk_amount = self.current_profile.account_balance * self.current_profile.risk_per_trade_pct
        
        stop_pips = abs(entry_price - stop_loss) * 10000
        if 'JPY' in symbol:
            stop_pips = abs(entry_price - stop_loss) * 100
        
        # Calculate lot size based on risk amount
        pip_value_per_lot = self.get_pip_value(symbol, 1.0)  # Pip value for 1 lot
        required_pip_value = risk_amount / stop_pips
        lot_size = required_pip_value / pip_value_per_lot
        lot_size = round(lot_size, 2)
        
        # Recalculate with actual lot size
        pip_value = self.get_pip_value(symbol, lot_size)
        risk_amount = stop_pips * pip_value
        
        # Target profit based on minimum R:R
        profit_pips = stop_pips * self.current_profile.min_risk_reward
        expected_profit = profit_pips * pip_value
        
        return lot_size, risk_amount, expected_profit
    
    def get_pip_value(self, symbol: str, lot_size: float) -> float:
        """Calculate pip value for a symbol and lot size"""
        # Simplified pip value calculation (assumes USD account)
        if 'JPY' in symbol:
            if symbol.startswith('USD'):
                return lot_size * 1000  # USDJPY: $1000 per lot per pip
            else:
                return lot_size * 1000  # Other JPY pairs approximation
        else:
            if symbol.endswith('USD'):
                return lot_size * 10  # EURUSD, GBPUSD: $10 per lot per pip
            else:
                return lot_size * 10  # Approximation for other pairs
    
    def calculate_position_size(self, method: PositionSizingMethod, symbol: str, entry_price: float, 
                              stop_loss: float, **kwargs) -> Tuple[float, float, float]:
        """
        Calculate lot size using specified method
        
        Returns:
            Tuple of (lot_size, risk_amount, expected_profit)
        """
        try:
            if method == PositionSizingMethod.MICRO_CONSERVATIVE:
                lot_size, risk_amount, expected_profit = self.calculate_micro_conservative_sizing(symbol, entry_price, stop_loss)
            
            elif method == PositionSizingMethod.MICRO_AGGRESSIVE:
                lot_size, risk_amount, expected_profit = self.calculate_micro_aggressive_sizing(symbol, entry_price, stop_loss)
            
            elif method == PositionSizingMethod.SMALL_STEADY:
                lot_size, risk_amount, expected_profit = self.calculate_small_steady_sizing(symbol, entry_price, stop_loss)
            
            elif method == PositionSizingMethod.PERCENTAGE_RISK:
                lot_size, risk_amount, expected_profit = self.calculate_percentage_risk_sizing(symbol, entry_price, stop_loss)
            
            else:
                # Fallback to conservative micro sizing
                lot_size, risk_amount, expected_profit = self.calculate_micro_conservative_sizing(symbol, entry_price, stop_loss)
            
            # Log the calculation
            self.log_position_sizing(symbol, method.value, entry_price, stop_loss, lot_size, risk_amount, expected_profit)
            
            return lot_size, risk_amount, expected_profit
            
        except Exception as e:
            self.logger.error(f"Position sizing calculation failed: {e}")
            return 0.01, 1.0, 1.0  # Safe fallback
    
    def log_position_sizing(self, symbol: str, method: str, entry_price: float, stop_loss: float, 
                           lot_size: float, risk_amount: float, expected_profit: float):
        """Log position sizing calculation to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO position_sizing_log 
                    (symbol, method, entry_price, stop_loss, calculated_lot_size, risk_amount, 
                     expected_profit, account_balance, calculation_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, method, entry_price, stop_loss, lot_size, risk_amount, expected_profit,
                    self.current_profile.account_balance, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log position sizing: {e}")
    
    def calculate_optimal_stops_and_targets(self, symbol: str, entry_price: float, direction: str) -> Tuple[float, float]:
        """
        Calculate realistic stop-loss and take-profit levels for small accounts
        """
        # Account tier-based stop distances (in pips)
        if self.current_profile.account_balance <= 200:
            # Micro accounts: tighter stops to preserve capital
            stop_pips = 15 if 'JPY' not in symbol else 15
            tp_multiplier = 1.5  # Conservative 1:1.5 R:R
        elif self.current_profile.account_balance <= 500:
            # Small micro accounts: slightly wider
            stop_pips = 20 if 'JPY' not in symbol else 20
            tp_multiplier = 2.0  # 1:2 R:R
        elif self.current_profile.account_balance <= 2000:
            # Small accounts: moderate stops
            stop_pips = 30 if 'JPY' not in symbol else 30
            tp_multiplier = 2.5  # 1:2.5 R:R
        else:
            # Larger accounts: wider stops
            stop_pips = 50 if 'JPY' not in symbol else 50
            tp_multiplier = 3.0  # 1:3 R:R
        
        # Convert pips to price
        if 'JPY' in symbol:
            pip_size = 0.01  # JPY pairs
        else:
            pip_size = 0.0001  # Most other pairs
        
        stop_distance = stop_pips * pip_size
        tp_distance = stop_distance * tp_multiplier
        
        if direction.lower() == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def recommend_position_sizing_method(self) -> PositionSizingMethod:
        """Recommend optimal position sizing method based on account size"""
        balance = self.current_profile.account_balance
        
        if balance <= 200:
            return PositionSizingMethod.MICRO_CONSERVATIVE
        elif balance <= 500:
            return PositionSizingMethod.MICRO_AGGRESSIVE
        elif balance <= 2000:
            return PositionSizingMethod.SMALL_STEADY
        else:
            return PositionSizingMethod.PERCENTAGE_RISK
    
    def calculate_growth_projection(self, win_rate: float, avg_profit: float, trades_per_month: int, months: int = 12) -> Dict:
        """Calculate realistic account growth projection"""
        current_balance = self.current_profile.account_balance
        monthly_results = []
        
        for month in range(months):
            wins = int(trades_per_month * win_rate)
            losses = trades_per_month - wins
            
            # Monthly profit/loss
            monthly_profit = (wins * avg_profit) - (losses * (avg_profit / 2))  # Assume avg loss is half avg profit
            current_balance += monthly_profit
            
            # Account tier progression
            if current_balance <= 500:
                tier = "MICRO"
            elif current_balance <= 2000:
                tier = "SMALL"
            elif current_balance <= 10000:
                tier = "MEDIUM"
            else:
                tier = "LARGE"
            
            monthly_results.append({
                'month': month + 1,
                'balance': current_balance,
                'tier': tier,
                'monthly_profit': monthly_profit,
                'total_trades': trades_per_month * (month + 1)
            })
        
        return {
            'starting_balance': self.current_profile.account_balance,
            'final_balance': current_balance,
            'total_growth': ((current_balance - self.current_profile.account_balance) / self.current_profile.account_balance) * 100,
            'monthly_breakdown': monthly_results
        }
    
    def generate_account_strategy_report(self) -> str:
        """Generate account-specific strategy recommendations"""
        balance = self.current_profile.account_balance
        recommended_method = self.recommend_position_sizing_method()
        
        if balance <= 200:
            strategy = f"""
╔══════════════════════════════════════════════════════════════╗
║                    MICRO ACCOUNT STRATEGY                    ║
║                     (${balance:.2f} Account)                     ║
╚══════════════════════════════════════════════════════════════╝

RECOMMENDED APPROACH: ULTRA CONSERVATIVE
├─ Position Size: 0.01 lots ONLY
├─ Risk per Trade: ${balance * 0.05:.2f} (5% of account)
├─ Profit Target: $0.50 - $2.00 per trade
├─ Stop Loss: 15-20 pips maximum
├─ Max Positions: 1-2 simultaneously
└─ Growth Target: 10-20% per month

STRATEGY FOCUS:
├─ Capital Preservation is #1 Priority
├─ Consistent small wins over big risks  
├─ Build experience with minimal risk
├─ Compound profits to reach $500 milestone
└─ Never risk more than $10 on a single trade

REALISTIC EXPECTATIONS:
├─ Monthly Profit: $10-40 (10-40% growth)
├─ Time to $500: 6-12 months
├─ Required Win Rate: 60%+ essential
└─ Trades per Month: 10-20 maximum
"""
        elif balance <= 500:
            strategy = f"""
╔══════════════════════════════════════════════════════════════╗
║                TRANSITIONAL MICRO STRATEGY                  ║
║                     (${balance:.2f} Account)                     ║
╚══════════════════════════════════════════════════════════════╝

RECOMMENDED APPROACH: CAUTIOUS AGGRESSIVE
├─ Position Size: 0.01-0.02 lots
├─ Risk per Trade: ${balance * 0.06:.2f} (6% of account)  
├─ Profit Target: $2.00 - $5.00 per trade
├─ Stop Loss: 20-25 pips
├─ Max Positions: 2-3 simultaneously
└─ Growth Target: 15-30% per month

STRATEGY FOCUS:
├─ Steady compounding with controlled risk
├─ Slightly more aggressive for faster growth
├─ Target $2000 to reach Small Account tier
├─ Begin diversification across 2-3 pairs
└─ Track performance metrics closely

REALISTIC EXPECTATIONS:
├─ Monthly Profit: $50-150 (15-30% growth)
├─ Time to $2000: 8-15 months
├─ Required Win Rate: 55%+ recommended
└─ Trades per Month: 15-30
"""
        else:
            strategy = f"""
╔══════════════════════════════════════════════════════════════╗
║                   SMALL ACCOUNT STRATEGY                    ║
║                     (${balance:.2f} Account)                    ║
╚══════════════════════════════════════════════════════════════╝

RECOMMENDED APPROACH: BALANCED GROWTH
├─ Position Size: 0.03-0.10 lots (progressive)
├─ Risk per Trade: ${balance * 0.03:.2f} (3% of account)
├─ Profit Target: $15.00 - $50.00 per trade
├─ Stop Loss: 25-40 pips
├─ Max Positions: 3-5 simultaneously  
└─ Growth Target: 10-25% per month

STRATEGY FOCUS:
├─ Consistent compounding with diversification
├─ Professional risk management principles
├─ Multiple currency pair exposure
├─ Target $10,000+ for Medium Account status
└─ Focus on risk-reward ratios 1:2.5+

REALISTIC EXPECTATIONS:
├─ Monthly Profit: $200-500 (10-25% growth)
├─ Time to $10,000: 12-24 months
├─ Required Win Rate: 50%+ sustainable
└─ Trades per Month: 20-40
"""
        
        return strategy
    
    def add_position(self, symbol: str, entry_price: float, direction: str, 
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                     method: Optional[PositionSizingMethod] = None) -> Position:
        """
        Add a new position with calculated sizing and check portfolio risk
        """
        if method is None:
            method = self.recommend_position_sizing_method()
        
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_stops_and_targets(symbol, entry_price, direction)
        
        lot_size, risk_amount, expected_profit = self.calculate_position_size(
            method, symbol, entry_price, stop_loss
        )
        
        # Check if new position would exceed portfolio risk
        if not self.can_add_position(risk_amount):
            self.logger.warning(f"Cannot add position: Would exceed max portfolio risk ({self.current_profile.max_portfolio_risk_pct*100}%)")
            raise ValueError("Portfolio risk limit exceeded")
        
        # Check max positions
        if len(self.positions) >= self.current_profile.max_positions:
            self.logger.warning("Max positions limit reached")
            raise ValueError("Maximum number of positions reached")
        
        # Validate risk-reward ratio
        actual_rr = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        if actual_rr < self.current_profile.min_risk_reward:
            self.logger.warning(f"Risk-reward ratio {actual_rr:.2f} below minimum {self.current_profile.min_risk_reward}")
        
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            entry_date=datetime.now(),
            risk_amount=risk_amount,
            expected_profit=expected_profit
        )
        
        self.positions.append(position)
        self.save_position_to_db(position)
        self.update_account_progression()
        
        self.logger.info(f"Added {direction} position for {symbol}: {lot_size} lots, Risk: ${risk_amount:.2f}")
        return position
    
    def can_add_position(self, proposed_risk: float) -> bool:
        """Check if a new position can be added without exceeding portfolio risk limits"""
        current_portfolio_risk = sum(p.risk_amount for p in self.positions)
        total_risk = current_portfolio_risk + proposed_risk
        max_allowed_risk = self.current_profile.account_balance * self.current_profile.max_portfolio_risk_pct
        
        return total_risk <= max_allowed_risk
    
    def save_position_to_db(self, position: Position):
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO positions 
                    (symbol, entry_price, lot_size, stop_loss, take_profit, direction, 
                     entry_date, risk_amount, expected_profit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position.symbol, position.entry_price, position.lot_size, position.stop_loss,
                    position.take_profit, position.direction, position.entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                    position.risk_amount, position.expected_profit
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save position to DB: {e}")
    
    def close_position(self, symbol: str, exit_price: float, actual_profit: float) -> bool:
        """Close a position and update account balance"""
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                # Update position in memory
                self.positions[i].exit_price = exit_price
                self.positions[i].status = 'closed'
                self.positions[i].exit_date = datetime.now()
                self.positions[i].actual_profit = actual_profit
                
                # Update account balance
                self.current_profile.account_balance += actual_profit
                
                # Save to DB
                self.update_position_in_db(symbol, exit_price, actual_profit)
                
                # Remove from active positions
                self.positions.pop(i)
                
                # Update progression
                self.update_account_progression()
                
                self.logger.info(f"Closed {symbol} at ${exit_price:.5f}, P&L: ${actual_profit:.2f}")
                return True
        
        self.logger.warning(f"Position {symbol} not found")
        return False
    
    def update_position_in_db(self, symbol: str, exit_price: float, actual_profit: float):
        """Update closed position in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE positions 
                    SET status = 'closed', exit_price = ?, exit_date = ?, actual_profit = ?
                    WHERE symbol = ? AND status = 'open'
                ''', (
                    exit_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), actual_profit, symbol
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update position in DB: {e}")
    
    def get_current_portfolio_risk(self) -> Dict[str, float]:
        """Get current portfolio risk metrics"""
        total_risk = sum(p.risk_amount for p in self.positions)
        risk_pct = (total_risk / self.current_profile.account_balance) * 100
        num_positions = len(self.positions)
        
        return {
            'total_risk_amount': total_risk,
            'risk_percentage': risk_pct,
            'num_positions': num_positions,
            'max_allowed_risk_pct': self.current_profile.max_portfolio_risk_pct * 100
        }
    
    def update_account_progression(self):
        """Update account progression tracking in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current stats from DB for accuracy
                df_closed = pd.read_sql_query('SELECT * FROM positions WHERE status = "closed"', conn)
                total_trades = len(df_closed)
                winning_trades = len(df_closed[df_closed['actual_profit'] > 0])
                total_profit = df_closed['actual_profit'].sum() if not df_closed.empty else 0.0
                
                # Calculate max drawdown (simplified - would need more historical data)
                max_drawdown = 0.0
                
                tier_name = "MICRO" if self.current_profile.account_balance <= 500 else \
                           "SMALL" if self.current_profile.account_balance <= 2000 else \
                           "MEDIUM" if self.current_profile.account_balance <= 10000 else "LARGE"
                
                conn.execute('''
                    INSERT INTO account_progression 
                    (date, account_balance, account_tier, total_trades, winning_trades, 
                     total_profit, max_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d'),
                    self.current_profile.account_balance, tier_name, total_trades, winning_trades,
                    total_profit, max_drawdown
                ))
                conn.commit()
                
                # Check for milestones
                self.check_growth_milestones()
                
        except Exception as e:
            self.logger.error(f"Failed to update account progression: {e}")
    
    def check_growth_milestones(self):
        """Check and log growth milestones"""
        milestones = [500, 1000, 2000, 5000, 10000, 25000]
        current_balance = self.current_profile.account_balance
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for milestone in milestones:
                    if current_balance >= milestone:
                        # Check if already achieved
                        cursor.execute('SELECT COUNT(*) FROM growth_milestones WHERE milestone_balance = ?', (milestone,))
                        if cursor.fetchone()[0] == 0:
                            # New milestone
                            tier_upgrade = self.get_tier_for_balance(milestone)
                            conn.execute('''
                                INSERT INTO growth_milestones 
                                (milestone_balance, achieved_date, tier_upgrade)
                                VALUES (?, ?, ?)
                            ''', (
                                milestone, datetime.now().strftime('%Y-%m-%d'), tier_upgrade
                            ))
                            self.logger.info(f"🎉 Milestone achieved: ${milestone} - Upgraded to {tier_upgrade} tier!")
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to check milestones: {e}")
    
    def get_tier_for_balance(self, balance: float) -> str:
        """Get tier name for a given balance"""
        if balance <= 500:
            return "MICRO"
        elif balance <= 2000:
            return "SMALL"
        elif balance <= 10000:
            return "MEDIUM"
        else:
            return "LARGE"
    
    def get_account_performance_summary(self) -> Dict:
        """Get a summary of account performance from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Recent progression
                df_progress = pd.read_sql_query('''
                    SELECT * FROM account_progression 
                    ORDER BY date DESC LIMIT 30
                ''', conn)
                
                if not df_progress.empty:
                    latest = df_progress.iloc[0]
                    win_rate = (latest['winning_trades'] / latest['total_trades'] * 100) if latest['total_trades'] > 0 else 0
                else:
                    latest = {}
                    win_rate = 0
                
                # Open positions summary
                df_positions = pd.read_sql_query('SELECT * FROM positions WHERE status = "open"', conn)
                open_risk = df_positions['risk_amount'].sum() if not df_positions.empty else 0
                
                return {
                    'current_balance': self.current_profile.account_balance,
                    'total_trades': latest.get('total_trades', 0),
                    'win_rate_pct': win_rate,
                    'total_profit': latest.get('total_profit', 0),
                    'max_drawdown': latest.get('max_drawdown', 0),
                    'open_positions': len(df_positions),
                    'open_risk_amount': open_risk
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def export_growth_data(self, format: str = 'csv') -> Optional[str]:
        """Export growth data to CSV or JSON"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('SELECT * FROM account_progression ORDER BY date ASC', conn)
                if format.lower() == 'csv':
                    filename = f'growth_export_{datetime.now().strftime("%Y%m%d")}.csv'
                    df.to_csv(filename, index=False)
                    return filename
                elif format.lower() == 'json':
                    filename = f'growth_export_{datetime.now().strftime("%Y%m%d")}.json'
                    df.to_json(filename, orient='records', date_format='iso')
                    return filename
                else:
                    return None
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize the risk manager
    rm = RiskManager()
    
    # Set account balance
    rm.set_account_profile(200.0)
    
    # Generate strategy report
    print(rm.generate_account_strategy_report())
    
    # Calculate growth projection
    projection = rm.calculate_growth_projection(win_rate=0.55, avg_profit=5.0, trades_per_month=20)
    print(f"Projected growth after 12 months: {projection['total_growth']:.1f}%")
    
    # Get performance summary
    summary = rm.get_account_performance_summary()
    print(summary)
    
    # Simulate adding a position
    try:
        position = rm.add_position("EURUSD", entry_price=1.0850, direction="long")
        print(f"Added position: {position}")
        
        # Check portfolio risk
        risk_info = rm.get_current_portfolio_risk()
        print(f"Current portfolio risk: {risk_info['risk_percentage']:.2f}%")
    except ValueError as e:
        print(f"Could not add position: {e}")
    
    # Export data
    exported_file = rm.export_growth_data('csv')
    if exported_file:
        print(f"Data exported to {exported_file}")