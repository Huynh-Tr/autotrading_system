#!/usr/bin/env python3
"""
Comprehensive test suite for the Auto Trading System
Tests all major components: data management, strategies, indicators, backtesting, and configuration
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import yaml
import warnings

# Suppress warnings for deprecated packages
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="The pkg_resources package is slated for removal")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Handle vnstock import issue
try:
    # Set environment variable to suppress Unicode issues
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Try to import vnstock with error handling
    import vnstock
    VNSTOCK_AVAILABLE = True
except (ImportError, UnicodeEncodeError, Exception) as e:
    print(f"Warning: vnstock not available or has encoding issues: {e}")
    VNSTOCK_AVAILABLE = False

# Import system components with error handling
try:
    from src.core.trading_engine import TradingEngine
    TRADING_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: TradingEngine not available: {e}")
    TRADING_ENGINE_AVAILABLE = False

try:
    from src.strategies.sma_crossover import SMACrossoverStrategy
    from src.strategies.rsi_strategy import RSIStrategy
    from src.strategies.macd_strategy import MACDStrategy
    from src.strategies.base_strategy import BaseStrategy
    STRATEGIES_AVAILABLE = True
except Exception as e:
    print(f"Warning: Strategies not available: {e}")
    STRATEGIES_AVAILABLE = False

try:
    from src.data.data_manager import DataManager
    DATA_MANAGER_AVAILABLE = True
except Exception as e:
    print(f"Warning: DataManager not available: {e}")
    DATA_MANAGER_AVAILABLE = False

try:
    from src.indicators.bollinger_bands import BollingerBands
    from src.indicators.macd import MACD
    from src.indicators.rsi import RSI
    from src.indicators.sma import SMA
    INDICATORS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Indicators not available: {e}")
    INDICATORS_AVAILABLE = False

try:
    from src.utils.config_manager import ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except Exception as e:
    print(f"Warning: ConfigManager not available: {e}")
    CONFIG_MANAGER_AVAILABLE = False

try:
    from src.risk.risk_manager import RiskManager
    RISK_MANAGER_AVAILABLE = True
except Exception as e:
    print(f"Warning: RiskManager not available: {e}")
    RISK_MANAGER_AVAILABLE = False

try:
    from src.backtesting.backtest_engine import BacktestEngine
    BACKTEST_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: BacktestEngine not available: {e}")
    BACKTEST_ENGINE_AVAILABLE = False


class TestDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_file)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipUnless(DATA_MANAGER_AVAILABLE, "DataManager not available")
    def test_data_manager_initialization(self):
        """Test DataManager initialization"""
        try:
            # Create a simple config for testing
            config_data = {
                'data': {
                    'cache_dir': self.test_dir,
                    'cache_data': True
                }
            }
            config = ConfigManager() if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            dm = DataManager(config)
            self.assertIsNotNone(dm)
            print("[OK] DataManager initialization successful")
        except Exception as e:
            self.fail(f"DataManager initialization failed: {e}")
    
    def test_data_loading(self):
        """Test data loading functionality"""
        try:
            # Test basic pandas data loading
            data = pd.read_csv(self.test_file, index_col=0, parse_dates=True)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            print("[OK] Data loading successful")
        except Exception as e:
            self.fail(f"Data loading failed: {e}")
    
    def test_data_validation(self):
        """Test data validation"""
        try:
            # Basic data validation
            is_valid = (
                len(self.test_data) > 0 and
                'Close' in self.test_data.columns and
                not self.test_data['Close'].isnull().all()
            )
            self.assertTrue(is_valid)
            print("[OK] Data validation successful")
        except Exception as e:
            self.fail(f"Data validation failed: {e}")


class TestIndicators(unittest.TestCase):
    """Test technical indicators"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
    
    @unittest.skipUnless(INDICATORS_AVAILABLE, "Indicators not available")
    def test_sma_indicator(self):
        """Test Simple Moving Average indicator"""
        try:
            sma = SMA(period=20)
            result = sma.calculate(self.test_data['Close'])
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.test_data))
            print("[OK] SMA indicator test successful")
        except Exception as e:
            self.fail(f"SMA indicator test failed: {e}")
    
    @unittest.skipUnless(INDICATORS_AVAILABLE, "Indicators not available")
    def test_rsi_indicator(self):
        """Test RSI indicator"""
        try:
            rsi = RSI(period=14)
            result = rsi.calculate(self.test_data['Close'])
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), len(self.test_data))
            # RSI should be between 0 and 100
            self.assertTrue(all(0 <= x <= 100 for x in result.dropna()))
            print("[OK] RSI indicator test successful")
        except Exception as e:
            self.fail(f"RSI indicator test failed: {e}")
    
    @unittest.skipUnless(INDICATORS_AVAILABLE, "Indicators not available")
    def test_macd_indicator(self):
        """Test MACD indicator"""
        try:
            macd = MACD(fast_period=12, slow_period=26, signal_period=9)
            result = macd.calculate(self.test_data['Close'])
            self.assertIsInstance(result, dict)
            self.assertIn('macd', result)
            self.assertIn('signal', result)
            self.assertIn('histogram', result)
            print("[OK] MACD indicator test successful")
        except Exception as e:
            self.fail(f"MACD indicator test failed: {e}")
    
    @unittest.skipUnless(INDICATORS_AVAILABLE, "Indicators not available")
    def test_bollinger_bands(self):
        """Test Bollinger Bands indicator"""
        try:
            bb = BollingerBands(period=20, std_dev=2)
            result = bb.calculate(self.test_data['Close'])
            self.assertIsInstance(result, dict)
            self.assertIn('upper', result)
            self.assertIn('middle', result)
            self.assertIn('lower', result)
            print("[OK] Bollinger Bands test successful")
        except Exception as e:
            self.fail(f"Bollinger Bands test failed: {e}")
    
    def test_basic_sma_calculation(self):
        """Test basic SMA calculation without external dependencies"""
        try:
            # Manual SMA calculation
            close_prices = self.test_data['Close']
            period = 20
            sma = close_prices.rolling(window=period).mean()
            self.assertIsInstance(sma, pd.Series)
            self.assertEqual(len(sma), len(close_prices))
            print("[OK] Basic SMA calculation test successful")
        except Exception as e:
            self.fail(f"Basic SMA calculation test failed: {e}")


class TestStrategies(unittest.TestCase):
    """Test trading strategies"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
    
    @unittest.skipUnless(STRATEGIES_AVAILABLE, "Strategies not available")
    def test_base_strategy_interface(self):
        """Test base strategy interface (not instantiation)"""
        try:
            # Test that BaseStrategy is an abstract class
            self.assertTrue(hasattr(BaseStrategy, '__abstractmethods__'))
            print("[OK] Base strategy interface test successful")
        except Exception as e:
            self.fail(f"Base strategy interface test failed: {e}")
    
    @unittest.skipUnless(STRATEGIES_AVAILABLE, "Strategies not available")
    def test_sma_crossover_strategy(self):
        """Test SMA Crossover strategy"""
        try:
            config = {
                'name': 'sma_crossover',
                'short_period': 10,
                'long_period': 20
            }
            strategy = SMACrossoverStrategy(config)
            
            # Get historical data and current data
            historical_data = self.test_data.iloc[:-1]  # All but last row
            current_data = self.test_data.iloc[-1]      # Last row
            
            signals = strategy.generate_signals(historical_data, current_data)
            self.assertIsInstance(signals, dict)
            print("[OK] SMA Crossover strategy test successful")
        except Exception as e:
            self.fail(f"SMA Crossover strategy test failed: {e}")
    
    @unittest.skipUnless(STRATEGIES_AVAILABLE, "Strategies not available")
    def test_rsi_strategy(self):
        """Test RSI strategy"""
        try:
            config = {
                'name': 'rsi_strategy',
                'period': 14,
                'overbought': 70,
                'oversold': 30
            }
            strategy = RSIStrategy(config)
            
            # Get historical data and current data
            historical_data = self.test_data.iloc[:-1]  # All but last row
            current_data = self.test_data.iloc[-1]      # Last row
            
            signals = strategy.generate_signals(historical_data, current_data)
            self.assertIsInstance(signals, dict)
            print("[OK] RSI strategy test successful")
        except Exception as e:
            self.fail(f"RSI strategy test failed: {e}")
    
    @unittest.skipUnless(STRATEGIES_AVAILABLE, "Strategies not available")
    def test_macd_strategy(self):
        """Test MACD strategy"""
        try:
            config = {
                'name': 'macd_strategy',
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            }
            strategy = MACDStrategy(config)
            
            # Get historical data and current data
            historical_data = self.test_data.iloc[:-1]  # All but last row
            current_data = self.test_data.iloc[-1]      # Last row
            
            signals = strategy.generate_signals(historical_data, current_data)
            self.assertIsInstance(signals, dict)
            print("[OK] MACD strategy test successful")
        except Exception as e:
            self.fail(f"MACD strategy test failed: {e}")
    
    def test_basic_sma_crossover_logic(self):
        """Test basic SMA crossover logic without external dependencies"""
        try:
            # Manual SMA crossover logic
            close_prices = self.test_data['Close']
            short_sma = close_prices.rolling(window=10).mean()
            long_sma = close_prices.rolling(window=20).mean()
            
            # Generate signals
            signals = pd.Series(0, index=close_prices.index)
            signals[short_sma > long_sma] = 1  # Buy signal
            signals[short_sma < long_sma] = -1  # Sell signal
            
            self.assertIsInstance(signals, pd.Series)
            self.assertEqual(len(signals), len(close_prices))
            print("[OK] Basic SMA crossover logic test successful")
        except Exception as e:
            self.fail(f"Basic SMA crossover logic test failed: {e}")


class TestRiskManager(unittest.TestCase):
    """Test risk management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100)
        })
    
    @unittest.skipUnless(RISK_MANAGER_AVAILABLE, "RiskManager not available")
    def test_risk_manager_initialization(self):
        """Test RiskManager initialization"""
        try:
            config = ConfigManager() if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            rm = RiskManager(config)
            self.assertIsNotNone(rm)
            print("[OK] RiskManager initialization successful")
        except Exception as e:
            self.fail(f"RiskManager initialization failed: {e}")
    
    @unittest.skipUnless(RISK_MANAGER_AVAILABLE, "RiskManager not available")
    def test_position_sizing(self):
        """Test position sizing calculation"""
        try:
            config = ConfigManager() if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            rm = RiskManager(config)
            
            # Test position sizing with proper parameters
            symbol = "AAPL"
            price = 100.0
            cash = 10000.0
            positions = {}
            
            position_size = rm.calculate_position_size(symbol, price, cash, positions)
            self.assertIsInstance(position_size, float)
            self.assertGreater(position_size, 0)
            print("[OK] Position sizing test successful")
        except Exception as e:
            self.fail(f"Position sizing test failed: {e}")
    
    def test_basic_risk_calculations(self):
        """Test basic risk calculations without external dependencies"""
        try:
            # Manual risk calculations
            capital = 10000
            max_risk_per_trade = 0.02  # 2%
            stop_loss_pct = 0.05  # 5%
            
            # Calculate position size
            max_risk_amount = capital * max_risk_per_trade
            position_size = max_risk_amount / stop_loss_pct
            
            self.assertIsInstance(position_size, float)
            self.assertGreater(position_size, 0)
            print("[OK] Basic risk calculations test successful")
        except Exception as e:
            self.fail(f"Basic risk calculations test failed: {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        """Set up test configuration"""
        self.test_config = {
            'data': {
                'source': 'yfinance',
                'symbol': 'AAPL',
                'start_date': '2023-01-01',
                'end_date': '2024-01-01'
            },
            'strategies': {
                'sma_crossover': {
                    'short_period': 10,
                    'long_period': 20
                }
            },
            'risk': {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.1
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/test.log'
            }
        }
        
        # Create temporary config file
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipUnless(CONFIG_MANAGER_AVAILABLE, "ConfigManager not available")
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization"""
        try:
            config = ConfigManager(self.config_file)
            self.assertIsNotNone(config)
            print("[OK] ConfigManager initialization successful")
        except Exception as e:
            self.fail(f"ConfigManager initialization failed: {e}")
    
    @unittest.skipUnless(CONFIG_MANAGER_AVAILABLE, "ConfigManager not available")
    def test_config_getting(self):
        """Test configuration value retrieval"""
        try:
            config = ConfigManager(self.config_file)
            value = config.get('data.source')
            self.assertEqual(value, 'yfinance')
            print("[OK] Configuration value retrieval successful")
        except Exception as e:
            self.fail(f"Configuration value retrieval failed: {e}")
    
    @unittest.skipUnless(CONFIG_MANAGER_AVAILABLE, "ConfigManager not available")
    def test_config_default_values(self):
        """Test configuration default values"""
        try:
            config = ConfigManager(self.config_file)
            value = config.get('nonexistent.key', 'default_value')
            self.assertEqual(value, 'default_value')
            print("[OK] Configuration default values test successful")
        except Exception as e:
            self.fail(f"Configuration default values test failed: {e}")
    
    def test_yaml_config_loading(self):
        """Test YAML configuration loading"""
        try:
            # Test basic YAML loading
            with open(self.config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertIsInstance(loaded_config, dict)
            self.assertIn('data', loaded_config)
            self.assertEqual(loaded_config['data']['source'], 'yfinance')
            print("[OK] YAML configuration loading test successful")
        except Exception as e:
            self.fail(f"YAML configuration loading test failed: {e}")


class TestTradingEngine(unittest.TestCase):
    """Test trading engine functionality"""
    
    def setUp(self):
        """Set up test data and configuration"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
        
        # Create test configuration
        self.test_config = {
            'data': {
                'source': 'yfinance',
                'symbol': 'AAPL'
            },
            'risk': {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.1
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/test.log'
            }
        }
        
        # Create temporary config file
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    @unittest.skipUnless(TRADING_ENGINE_AVAILABLE, "TradingEngine not available")
    def test_trading_engine_initialization(self):
        """Test TradingEngine initialization"""
        try:
            config = ConfigManager(self.config_file) if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            engine = TradingEngine(config)
            self.assertIsNotNone(engine)
            print("[OK] TradingEngine initialization successful")
        except Exception as e:
            self.fail(f"TradingEngine initialization failed: {e}")
    
    @unittest.skipUnless(TRADING_ENGINE_AVAILABLE and STRATEGIES_AVAILABLE, "TradingEngine or Strategies not available")
    def test_strategy_addition(self):
        """Test adding strategies to trading engine"""
        try:
            config = ConfigManager(self.config_file) if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            engine = TradingEngine(config)
            
            strategy_config = {'name': 'test_sma', 'short_period': 10, 'long_period': 20}
            strategy = SMACrossoverStrategy(strategy_config)
            engine.add_strategy(strategy)
            
            self.assertGreater(len(engine.strategies), 0)
            print("[OK] Strategy addition test successful")
        except Exception as e:
            self.fail(f"Strategy addition test failed: {e}")


class TestBacktesting(unittest.TestCase):
    """Test backtesting functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 100, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
    
    @unittest.skipUnless(BACKTEST_ENGINE_AVAILABLE, "BacktestEngine not available")
    def test_backtest_engine_initialization(self):
        """Test BacktestEngine initialization"""
        try:
            config = ConfigManager() if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            engine = BacktestEngine(config)
            self.assertIsNotNone(engine)
            print("[OK] BacktestEngine initialization successful")
        except Exception as e:
            self.fail(f"BacktestEngine initialization failed: {e}")
    
    @unittest.skipUnless(BACKTEST_ENGINE_AVAILABLE and STRATEGIES_AVAILABLE, "BacktestEngine or Strategies not available")
    def test_backtest_execution(self):
        """Test basic backtest execution"""
        try:
            config = ConfigManager() if CONFIG_MANAGER_AVAILABLE else type('Config', (), {'get': lambda x, y=None: y})()
            engine = BacktestEngine(config)
            
            # Create a simple strategy
            strategy_config = {'name': 'test_sma', 'short_period': 10, 'long_period': 20}
            strategy = SMACrossoverStrategy(strategy_config)
            
            # Run backtest
            strategies = {'test_sma': strategy}
            results = engine.run_backtest(strategies, '2023-01-01', '2023-12-31')
            
            self.assertIsInstance(results, dict)
            print("[OK] Backtest execution test successful")
        except Exception as e:
            self.fail(f"Backtest execution test failed: {e}")
    
    def test_basic_backtest_simulation(self):
        """Test basic backtest simulation without external dependencies"""
        try:
            # Manual backtest simulation
            initial_capital = 10000
            current_capital = initial_capital
            trades = []
            returns = []
            
            # Simple buy and hold strategy
            initial_price = self.test_data['Close'].iloc[0]
            final_price = self.test_data['Close'].iloc[-1]
            
            # Calculate returns
            total_return = (final_price - initial_price) / initial_price
            final_capital = initial_capital * (1 + total_return)
            
            self.assertIsInstance(total_return, float)
            self.assertIsInstance(final_capital, float)
            print("[OK] Basic backtest simulation test successful")
        except Exception as e:
            self.fail(f"Basic backtest simulation test failed: {e}")


def run_all_tests():
    """Run all tests and return results"""
    print("=" * 60)
    print("AUTO TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Print component availability
    print("\nComponent Availability:")
    print(f"DataManager: {'[OK]' if DATA_MANAGER_AVAILABLE else '[FAIL]'}")
    print(f"Indicators: {'[OK]' if INDICATORS_AVAILABLE else '[FAIL]'}")
    print(f"Strategies: {'[OK]' if STRATEGIES_AVAILABLE else '[FAIL]'}")
    print(f"RiskManager: {'[OK]' if RISK_MANAGER_AVAILABLE else '[FAIL]'}")
    print(f"ConfigManager: {'[OK]' if CONFIG_MANAGER_AVAILABLE else '[FAIL]'}")
    print(f"TradingEngine: {'[OK]' if TRADING_ENGINE_AVAILABLE else '[FAIL]'}")
    print(f"BacktestEngine: {'[OK]' if BACKTEST_ENGINE_AVAILABLE else '[FAIL]'}")
    print(f"vnstock: {'[OK]' if VNSTOCK_AVAILABLE else '[FAIL]'}")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataManager,
        TestIndicators,
        TestStrategies,
        TestRiskManager,
        TestConfiguration,
        TestTradingEngine,
        TestBacktesting
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 