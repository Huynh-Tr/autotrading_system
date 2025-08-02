# Quick Backtest Guide

## 🚀 Cách chạy Backtest nhanh trong Notebook

### Phương pháp 1: Sử dụng Demo Script

```python
# Import demo script
from backtest_demo import run_backtest_demo

# Chạy full demo
results = run_backtest_demo()
```

### Phương pháp 2: Quick Backtest

```python
# Import quick function
from backtest_demo import quick_backtest

# Chạy backtest nhanh
engine = quick_backtest()
```

### Phương pháp 3: Manual Setup

```python
# 1. Setup imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy

# 2. Khởi tạo engine
engine = TradingEngine("../../config/config.yaml")

# 3. Thêm strategy
sma_strategy = SMACrossoverStrategy(short_window=10, long_window=30, name="SMA_Crossover")
engine.add_strategy(sma_strategy)

# 4. Chạy backtest
engine.run_backtest("2023-01-01", "2023-12-31")

# 5. Hiển thị kết quả
portfolio_summary = engine.get_portfolio_summary()
print(f"Final Value: ${portfolio_summary['total_value']:,.2f}")
print(f"Total Return: {portfolio_summary['total_return']:.2%}")
```

## 📊 Kết quả sẽ hiển thị:

- **Portfolio Value**: Giá trị portfolio theo thời gian
- **Total Return**: Tổng lợi nhuận
- **Sharpe Ratio**: Tỷ lệ Sharpe
- **Max Drawdown**: Drawdown tối đa
- **Win Rate**: Tỷ lệ thắng
- **Total Trades**: Tổng số trades

## 🔧 Troubleshooting

### Lỗi Import
```python
# Kiểm tra sys.path
print("Current working directory:", os.getcwd())
print("Project root:", project_root)
```

### Lỗi Config
```python
# Kiểm tra config
print("Config symbols:", engine.config.get('trading.symbols'))
print("Config date range:", engine.config.get('data.start_date'), "-", engine.config.get('data.end_date'))
```

### Lỗi Data
```python
# Kiểm tra data availability
from src.data.data_manager import DataManager
data_manager = DataManager(engine.config)
print("Available symbols:", data_manager.get_available_symbols())
```

## 📁 Files quan trọng:

- `backtest_demo.py`: Demo script hoàn chỉnh
- `BACKTEST_GUIDE.md`: Hướng dẫn chi tiết
- `config/config.yaml`: Cấu hình hệ thống 