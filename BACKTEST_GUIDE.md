# Hướng dẫn Chạy Backtest và Hiển thị Kết quả

## Phân tích Notebook hiện tại

Sau khi kiểm tra notebook `trading.ipynb`, tôi phát hiện những vấn đề sau cần điều chỉnh:

### 🔍 Vấn đề hiện tại:

1. **Import không đúng**: Notebook đang import `TradingViewData` từ thư mục gốc thay vì từ `src`
2. **Thiếu kết nối với TradingEngine**: Chưa sử dụng TradingEngine để chạy backtest
3. **Data format không tương thích**: Data từ TradingView không đúng format cho TradingEngine
4. **Thiếu phần chạy backtest**: Chưa có code để thực sự chạy backtest
5. **Thiếu hiển thị kết quả**: Chưa có code để visualize kết quả

## 🔧 Điều chỉnh cần thiết

### 1. Sửa Import và Setup

```python
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
```

### 2. Khởi tạo TradingEngine

```python
# Cell 2: Khởi tạo TradingEngine
config_path = "../../config/config.yaml"
engine = TradingEngine(config_path)

print(f"TradingEngine đã được khởi tạo với vốn ban đầu: ${engine.cash:,.2f}")
print(f"Config loaded: {engine.config.get('trading.symbols')}")
```

### 3. Chuẩn bị Data cho Backtest

```python
# Cell 3: Chuẩn bị data
# Sử dụng data có sẵn hoặc lấy từ DataManager
symbols = engine.config.get("trading.symbols", ["AAPL"])
start_date = engine.config.get("data.start_date", "2023-01-01")
end_date = engine.config.get("data.end_date", "2023-12-31")

print(f"Lấy data cho symbols: {symbols}")
print(f"Period: {start_date} đến {end_date}")

# Lấy data từ DataManager
data_manager = DataManager(engine.config)
historical_data = data_manager.get_historical_data(
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    interval=engine.config.get("data.interval", "1d")
)

print(f"Data shape: {historical_data.shape}")
print(f"Data columns: {historical_data.columns.tolist()}")
print(f"Date range: {historical_data.index.min()} đến {historical_data.index.max()}")
```

### 4. Thêm Strategies

```python
# Cell 4: Thêm strategies
# SMA Crossover Strategy
sma_strategy = SMACrossoverStrategy(
    short_window=10,
    long_window=30,
    name="SMA_Crossover"
)
engine.add_strategy(sma_strategy)

# RSI Strategy
rsi_strategy = RSIStrategy(
    period=14,
    oversold_threshold=30,
    overbought_threshold=70,
    name="RSI_Strategy"
)
engine.add_strategy(rsi_strategy)

# MACD Strategy
macd_strategy = MACDStrategy(
    fast_period=12,
    slow_period=26,
    signal_period=9,
    name="MACD_Strategy"
)
engine.add_strategy(macd_strategy)

print(f"Đã thêm {len(engine.strategies)} strategies:")
for name, strategy in engine.strategies.items():
    print(f"  - {name}: {strategy.__class__.__name__}")
```

### 5. Chạy Backtest

```python
# Cell 5: Chạy backtest
print("Bắt đầu chạy backtest...")
print(f"Period: {start_date} đến {end_date}")

# Chạy backtest
engine.run_backtest(start_date, end_date)

print("Backtest hoàn thành!")
```

### 6. Hiển thị Kết quả

```python
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
```

### 7. Visualize Kết quả

```python
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
```

### 8. Chi tiết Trades

```python
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
```

### 9. Strategy Performance

```python
# Cell 9: So sánh performance của các strategies
def compare_strategies(engine):
    """Compare performance of different strategies"""
    if not hasattr(engine, 'strategies') or not engine.strategies:
        print("Không có strategies để so sánh")
        return
    
    strategy_results = {}
    
    for name, strategy in engine.strategies.items():
        # Calculate basic metrics for each strategy
        # This is a simplified version - you might want to run separate backtests
        strategy_results[name] = {
            'name': name,
            'type': strategy.__class__.__name__,
            'parameters': strategy.get_summary()
        }
    
    # Display strategy comparison
    print("=== STRATEGY COMPARISON ===")
    for name, result in strategy_results.items():
        print(f"\nStrategy: {name}")
        print(f"Type: {result['type']}")
        print(f"Parameters: {result['parameters']}")

compare_strategies(engine)
```

## 🚀 Cách sử dụng trong Notebook

1. **Copy các cells trên vào notebook**
2. **Chạy từng cell theo thứ tự**
3. **Kiểm tra kết quả và điều chỉnh parameters nếu cần**

## ⚠️ Lưu ý quan trọng

1. **Data Source**: Đảm bảo data source trong config.yaml phù hợp
2. **Date Range**: Kiểm tra start_date và end_date trong config
3. **Symbols**: Đảm bảo symbols trong config có data available
4. **Dependencies**: Cài đặt tất cả required packages
5. **Config File**: Đảm bảo config.yaml tồn tại và đúng format

## 🔧 Troubleshooting

### Lỗi ImportError
```python
# Kiểm tra sys.path
print("Current sys.path:")
for path in sys.path:
    print(f"  {path}")
```

### Lỗi Data không tìm thấy
```python
# Kiểm tra data availability
data_manager = DataManager(engine.config)
available_data = data_manager.get_available_symbols()
print(f"Available symbols: {available_data}")
```

### Lỗi Config
```python
# Kiểm tra config
print("Config loaded:")
print(f"  Symbols: {engine.config.get('trading.symbols')}")
print(f"  Initial Capital: {engine.config.get('trading.initial_capital')}")
print(f"  Date Range: {engine.config.get('data.start_date')} - {engine.config.get('data.end_date')}")
``` 