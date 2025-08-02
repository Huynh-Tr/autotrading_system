# Cell 1: Setup imports
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import TradingEngine và components
from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager
from src.data.data_manager import DataManager

# Import visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

print("Import thành công!")

# Cell 2: Khởi tạo TradingEngine
config_path = "../../config/config.yaml"
engine = TradingEngine(config_path)

print(f"TradingEngine đã được khởi tạo với vốn ban đầu: ${engine.cash:,.2f}")
print(f"Config loaded: {engine.config.get('trading.symbols')}")
print(f"Config loaded: {engine.config.get('data.source')}")

# Cell 3: Chuẩn bị data
# Sử dụng data có sẵn hoặc lấy từ DataManager
symbols = engine.config.get("trading.symbols", ["AAPL"])
start_date = engine.config.get("data.start_date", "2023-01-01")
end_date = engine.config.get("data.end_date", "2023-12-31")


# Lấy data từ DataManager
data_manager = DataManager(engine.config)
historical_data = data_manager.get_historical_data(
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    interval=engine.config.get("data.interval", "1d"),
    n_bars=engine.config.get("data.n_bars", 1000)
)

print(f"Lấy data cho symbols: {symbols}")
# print(f"Period: {start_date} đến {end_date}")
print(f"Data shape: {historical_data.shape}")
print(f"Data columns: {historical_data.columns.tolist()}")
print(
    f"Date range: {historical_data.index.min()} \
        đến {historical_data.index.max()} \
            interval: {engine.config.get('data.interval', '1d')}"
)

# Cell 4: Thêm strategies
# SMA Crossover Strategy
sma_strategy = SMACrossoverStrategy({
    "short_window": engine.config.get("strategies.sma_crossover.short_window", 5),
    "long_window": engine.config.get("strategies.sma_crossover.long_window", 30),
    "name": "SMA_Crossover"
})
engine.add_strategy(sma_strategy)

# RSI Strategy
rsi_strategy = RSIStrategy({
    "period": engine.config.get("strategies.rsi.period", 14),
    "oversold": engine.config.get("strategies.rsi.oversold", 30),
    "overbought": engine.config.get("strategies.rsi.overbought", 70),
    "name": "RSI_Strategy"
})
engine.add_strategy(rsi_strategy)

# MACD Strategy
macd_strategy = MACDStrategy({
    "fast_period": engine.config.get("strategies.macd.fast_period", 15),
    "slow_period": engine.config.get("strategies.macd.slow_period", 20),
    "signal_period": engine.config.get("strategies.macd.signal_period", 7),
    "name": "MACD_Strategy"
})
engine.add_strategy(macd_strategy)

print(f"Đã thêm {len(engine.strategies)} strategies:")
for name, strategy in engine.strategies.items():
    print(f"  - {name}: {strategy.__class__.__name__}")

# Cell 5: Chạy backtest
print("Bắt đầu chạy backtest...")

# Chạy chiến lược
print('SMA Crossover:', engine.config.get('strategies.sma_crossover.enabled', False))
print('RSI:', engine.config.get('strategies.rsi.enabled', False))
print('MACD:', engine.config.get('strategies.macd.enabled', False), '\n')

# Chạy backtest với chiến lược đã chọn
engine.run_backtest(start_date, end_date)

# Cell 6: Lấy kết quả backtest
portfolio_summary = engine.get_portfolio_summary()

print("=== KẾT QUẢ BACKTEST ===")
print(f"Initial Capital: ${portfolio_summary['initial_capital']:,.2f}")
print(f"Final Portfolio Value: ${portfolio_summary['total_value']:,.2f}")
print(f"Total Return: {portfolio_summary['total_return']:.2%}")
print(f"Annualized Return: {portfolio_summary['annualized_return']:.2%}")
print(f"Sharpe Ratio: {portfolio_summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {portfolio_summary['max_drawdown']:.2%}")
print(f"Win Rate: {portfolio_summary['win_rate']:.2%}")
print(f"Total Trades: {portfolio_summary['total_trades']}")

# Cell 7: Visualize portfolio performance
def plot_portfolio_performance(engine):
    """Plot portfolio performance"""
    portfolio_history = engine.portfolio_history
    
    if not portfolio_history:
        print("Không có dữ liệu portfolio history")
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
                  mode='lines', name='Portfolio Value'),
        row=1, col=1
    )
    
    # Daily Returns
    daily_returns = df['total_value'].pct_change()
    fig.add_trace(
        go.Scatter(x=df.index, y=daily_returns, 
                  mode='lines', name='Daily Returns'),
        row=1, col=2
    )
    
    # Cumulative Returns
    cumulative_returns = (1 + daily_returns).cumprod()
    fig.add_trace(
        go.Scatter(x=df.index, y=cumulative_returns, 
                  mode='lines', name='Cumulative Returns'),
        row=2, col=1
    )
    
    # Drawdown
    running_max = df['total_value'].expanding().max()
    drawdown = (df['total_value'] - running_max) / running_max
    fig.add_trace(
        go.Scatter(x=df.index, y=drawdown, 
                  mode='lines', name='Drawdown', fill='tonexty'),
        row=2, col=2
    )
    
    # Cash vs Positions
    fig.add_trace(
        go.Scatter(x=df.index, y=df['cash'], 
                  mode='lines', name='Cash'),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['total_value'] - df['cash'], 
                  mode='lines', name='Positions'),
        row=3, col=1
    )
    
    # Trade Distribution (if available)
    if hasattr(engine, 'trades') and engine.trades:
        trade_returns = [trade.pnl for trade in engine.trades]
        fig.add_trace(
            go.Histogram(x=trade_returns, name='Trade Returns'),
            row=3, col=2
        )
    
    fig.update_layout(height=900, title_text="Portfolio Performance Analysis")
    fig.show()

# Chạy visualization
plot_portfolio_performance(engine)

# Cell 8: Hiển thị chi tiết trades
if hasattr(engine, 'trades') and engine.trades:
    print(f"\n=== CHI TIẾT TRADES ({len(engine.trades)} trades) ===")
    
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
        for trade in engine.trades
    ])
    
    print(trades_df.head(10))
    
    # Trade statistics
    print(f"\n=== TRADE STATISTICS ===")
    print(f"Total Trades: {len(engine.trades)}")
    print(f"Buy Trades: {len(trades_df[trades_df['Side'] == 'buy'])}")
    print(f"Sell Trades: {len(trades_df[trades_df['Side'] == 'sell'])}")
    print(f"Average Trade Size: {trades_df['Quantity'].mean():.2f}")
    print(f"Total Commission: ${trades_df['Commission'].sum():.2f}")
else:
    print("Không có trades nào được thực hiện")

# Cell 9: Visualize trades
def viz(symbol, trades_df, historical_data):
    # Visualize
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=historical_data[symbol].index,
                    open=historical_data[symbol].open,
                    high=historical_data[symbol].high,
                    low=historical_data[symbol].low,
                    close=historical_data[symbol].close,
                    ), row=1, col=1)
    set_ylim = (historical_data[symbol].low.min() * 0.98, historical_data[symbol].high.max() * 1.02)
    # Add buy/sell markers from trades_df to the candlestick chart (row 1)
    if 'trades_df' in locals():
        buy_trades = trades_df[trades_df['Side'] == 'buy']
        sell_trades = trades_df[trades_df['Side'] == 'sell']
        # Buy markers
        fig.add_trace(
            go.Scatter(
                x=buy_trades['Timestamp'],
                y=historical_data[symbol][historical_data.index.isin(buy_trades['Timestamp'])].low * 0.998,
                mode='markers+text',
                marker=dict(symbol='triangle-up', color='green', size=12),
                text=['Buy']*len(buy_trades),
                textposition='bottom center',
                name='Buy'
            ),
            row=1, col=1
        )
        # Sell markers
        fig.add_trace(
            go.Scatter(
                x=sell_trades['Timestamp'],
                y=historical_data[symbol][historical_data.index.isin(sell_trades['Timestamp'])].high * 1.002,
                mode='markers+text',
                marker=dict(symbol='triangle-down', color='red', size=12),
                text=['Sell']*len(sell_trades),
                textposition='top center',
                name='Sell'
            ),
            row=1, col=1
        )

    fig.add_trace(go.Bar(x=historical_data[symbol].index,
                        y=historical_data[symbol].volume,
                        ), row=2, col=1)

    fig.update_layout(title=f'{symbol}',
                    yaxis_range=(set_ylim[0], set_ylim[1]),
                    xaxis_title='Date',
                    yaxis_title='Price', 
                    height=800, width=1000)
    fig.show()

viz(symbols[0], trades_df, historical_data)