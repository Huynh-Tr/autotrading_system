"""
Demo script để chạy backtest và hiển thị kết quả
File này có thể được import vào notebook hoặc chạy độc lập
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Setup imports
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager
from src.data.data_manager import DataManager

class BacktestDemo:
    """Demo class để chạy backtest và hiển thị kết quả"""
    
    def __init__(self, config_path="../../config/config.yaml"):
        """Khởi tạo demo"""
        self.config_path = config_path
        self.engine = None
        self.results = {}
        
    def setup_engine(self):
        """Khởi tạo TradingEngine"""
        print("=== KHỞI TẠO TRADING ENGINE ===")
        
        self.engine = TradingEngine(self.config_path)
        
        print(f"✅ TradingEngine đã được khởi tạo")
        print(f"💰 Initial Capital: ${self.engine.cash:,.2f}")
        print(f"📊 Symbols: {self.engine.config.get('trading.symbols')}")
        print(f"📅 Date Range: {self.engine.config.get('data.start_date')} - {self.engine.config.get('data.end_date')}")
        
        return self.engine
    
    def add_strategies(self):
        """Thêm các strategies"""
        print("\n=== THÊM STRATEGIES ===")
        
        # SMA Crossover Strategy
        sma_config = {
            "short_window": 10,
            "long_window": 30,
            "name": "SMA_Crossover"
        }
        sma_strategy = SMACrossoverStrategy(sma_config)
        self.engine.add_strategy(sma_strategy)
        
        # RSI Strategy
        rsi_config = {
            "period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "name": "RSI_Strategy"
        }
        rsi_strategy = RSIStrategy(rsi_config)
        self.engine.add_strategy(rsi_strategy)
        
        # MACD Strategy
        macd_config = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "name": "MACD_Strategy"
        }
        macd_strategy = MACDStrategy(macd_config)
        self.engine.add_strategy(macd_strategy)
        
        print(f"✅ Đã thêm {len(self.engine.strategies)} strategies:")
        for name, strategy in self.engine.strategies.items():
            print(f"   - {name}: {strategy.__class__.__name__}")
        
        return self.engine.strategies
    
    def prepare_data(self):
        """Chuẩn bị data cho backtest"""
        print("\n=== CHUẨN BỊ DATA ===")
        
        symbols = self.engine.config.get("trading.symbols", ["AAPL"])
        start_date = self.engine.config.get("data.start_date", "2023-01-01")
        end_date = self.engine.config.get("data.end_date", "2023-12-31")
        
        print(f"📈 Symbols: {symbols}")
        print(f"📅 Period: {start_date} đến {end_date}")
        
        # Lấy data từ DataManager
        data_manager = DataManager(self.engine.config)
        historical_data = data_manager.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=self.engine.config.get("data.interval", "1d")
        )
        
        print(f"✅ Data loaded: {historical_data.shape}")
        print(f"📊 Columns: {historical_data.columns.tolist()}")
        print(f"📅 Date range: {historical_data.index.min()} đến {historical_data.index.max()}")
        
        return historical_data
    
    def run_backtest(self):
        """Chạy backtest"""
        print("\n=== CHẠY BACKTEST ===")
        
        start_date = self.engine.config.get("data.start_date", "2023-01-01")
        end_date = self.engine.config.get("data.end_date", "2023-12-31")
        
        print(f"🚀 Bắt đầu backtest từ {start_date} đến {end_date}")
        
        # Chạy backtest
        self.engine.run_backtest(start_date, end_date)
        
        print("✅ Backtest hoàn thành!")
        
        return self.engine
    
    def display_results(self):
        """Hiển thị kết quả backtest"""
        print("\n=== KẾT QUẢ BACKTEST ===")
        
        portfolio_summary = self.engine.get_portfolio_summary()
        
        # Hiển thị kết quả chính
        print(f"💰 Initial Capital: ${portfolio_summary['initial_capital']:,.2f}")
        print(f"💵 Final Portfolio Value: ${portfolio_summary['total_value']:,.2f}")
        print(f"📈 Total Return: {portfolio_summary['total_return']:.2%}")
        print(f"📊 Annualized Return: {portfolio_summary['annualized_return']:.2%}")
        print(f"⚡ Sharpe Ratio: {portfolio_summary['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {portfolio_summary['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {portfolio_summary['win_rate']:.2%}")
        print(f"🔄 Total Trades: {portfolio_summary['total_trades']}")
        
        return portfolio_summary
    
    def plot_portfolio_performance(self):
        """Vẽ biểu đồ performance"""
        print("\n=== VẼ BIỂU ĐỒ PERFORMANCE ===")
        
        portfolio_history = self.engine.portfolio_history
        
        if not portfolio_history:
            print("❌ Không có dữ liệu portfolio history")
            return
        
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Daily Returns', 'Cumulative Returns', 
                           'Drawdown', 'Cash vs Positions', 'Trade Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio Value
        fig.add_trace(
            go.Scatter(x=df.index, y=df['total_value'], 
                      mode='lines', name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Daily Returns
        daily_returns = df['total_value'].pct_change()
        fig.add_trace(
            go.Scatter(x=df.index, y=daily_returns, 
                      mode='lines', name='Daily Returns', line=dict(color='green')),
            row=1, col=2
        )
        
        # Cumulative Returns
        cumulative_returns = (1 + daily_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=df.index, y=cumulative_returns, 
                      mode='lines', name='Cumulative Returns', line=dict(color='red')),
            row=2, col=1
        )
        
        # Drawdown
        running_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=df.index, y=drawdown, 
                      mode='lines', name='Drawdown', fill='tonexty', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Cash vs Positions
        fig.add_trace(
            go.Scatter(x=df.index, y=df['cash'], 
                      mode='lines', name='Cash', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['total_value'] - df['cash'], 
                      mode='lines', name='Positions', line=dict(color='brown')),
            row=3, col=1
        )
        
        # Trade Distribution (if available)
        if hasattr(self.engine, 'trades') and self.engine.trades:
            trade_returns = [trade.pnl for trade in self.engine.trades]
            fig.add_trace(
                go.Histogram(x=trade_returns, name='Trade Returns', marker_color='lightblue'),
                row=3, col=2
            )
        
        fig.update_layout(
            height=900, 
            title_text="Portfolio Performance Analysis",
            showlegend=True
        )
        
        fig.show()
        
        print("✅ Biểu đồ performance đã được hiển thị")
    
    def display_trades(self):
        """Hiển thị chi tiết trades"""
        print("\n=== CHI TIẾT TRADES ===")
        
        if hasattr(self.engine, 'trades') and self.engine.trades:
            print(f"📊 Tổng số trades: {len(self.engine.trades)}")
            
            trades_df = pd.DataFrame([
                {
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Quantity': trade.quantity,
                    'Price': trade.price,
                    'Timestamp': trade.timestamp,
                    'Commission': trade.commission,
                    'Strategy': trade.strategy
                }
                for trade in self.engine.trades
            ])
            
            print("\n📋 10 trades đầu tiên:")
            print(trades_df.head(10).to_string(index=False))
            
            # Trade statistics
            print(f"\n📈 TRADE STATISTICS:")
            print(f"   Total Trades: {len(self.engine.trades)}")
            print(f"   Buy Trades: {len(trades_df[trades_df['Side'] == 'buy'])}")
            print(f"   Sell Trades: {len(trades_df[trades_df['Side'] == 'sell'])}")
            print(f"   Average Trade Size: {trades_df['Quantity'].mean():.2f}")
            print(f"   Total Commission: ${trades_df['Commission'].sum():.2f}")
            
            return trades_df
        else:
            print("❌ Không có trades nào được thực hiện")
            return None
    
    def compare_strategies(self):
        """So sánh performance của các strategies"""
        print("\n=== SO SÁNH STRATEGIES ===")
        
        if not hasattr(self.engine, 'strategies') or not self.engine.strategies:
            print("❌ Không có strategies để so sánh")
            return
        
        strategy_results = {}
        
        for name, strategy in self.engine.strategies.items():
            strategy_results[name] = {
                'name': name,
                'type': strategy.__class__.__name__,
                'parameters': strategy.get_summary()
            }
        
        print("📊 Strategy Comparison:")
        for name, result in strategy_results.items():
            print(f"\n   Strategy: {name}")
            print(f"   Type: {result['type']}")
            print(f"   Parameters: {result['parameters']}")
        
        return strategy_results
    
    def run_full_demo(self):
        """Chạy toàn bộ demo"""
        print("🚀 BẮT ĐẦU BACKTEST DEMO")
        print("=" * 50)
        
        try:
            # 1. Setup engine
            self.setup_engine()
            
            # 2. Add strategies
            self.add_strategies()
            
            # 3. Prepare data
            self.prepare_data()
            
            # 4. Run backtest
            self.run_backtest()
            
            # 5. Display results
            results = self.display_results()
            
            # 6. Plot performance
            self.plot_portfolio_performance()
            
            # 7. Display trades
            trades_df = self.display_trades()
            
            # 8. Compare strategies
            strategy_results = self.compare_strategies()
            
            print("\n" + "=" * 50)
            print("✅ BACKTEST DEMO HOÀN THÀNH!")
            
            return {
                'engine': self.engine,
                'results': results,
                'trades': trades_df,
                'strategies': strategy_results
            }
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình chạy demo: {e}")
            import traceback
            traceback.print_exc()
            return None

# Hàm tiện ích để chạy demo
def run_backtest_demo(config_path="../../config/config.yaml"):
    """Chạy backtest demo"""
    demo = BacktestDemo(config_path)
    return demo.run_full_demo()

# Hàm để chạy từ notebook
def quick_backtest():
    """Chạy backtest nhanh cho notebook"""
    print("🚀 Quick Backtest Demo")
    
    # Setup
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import
    from src.core.trading_engine import TradingEngine
    from src.strategies.sma_crossover import SMACrossoverStrategy
    
    # Khởi tạo engine
    engine = TradingEngine("../../config/config.yaml")
    
    # Thêm strategy với config đúng
    sma_config = {
        "short_window": 10,
        "long_window": 30,
        "name": "SMA_Crossover"
    }
    sma_strategy = SMACrossoverStrategy(sma_config)
    engine.add_strategy(sma_strategy)
    
    # Chạy backtest
    engine.run_backtest("2023-01-01", "2023-12-31")
    
    # Hiển thị kết quả
    portfolio_summary = engine.get_portfolio_summary()
    print(f"💰 Final Value: ${portfolio_summary['total_value']:,.2f}")
    print(f"📈 Total Return: {portfolio_summary['total_return']:.2%}")
    print(f"🔄 Total Trades: {portfolio_summary['total_trades']}")
    
    return engine

if __name__ == "__main__":
    # Chạy demo khi chạy file trực tiếp
    results = run_backtest_demo()
    if results:
        print("Demo completed successfully!")
    else:
        print("Demo failed!") 