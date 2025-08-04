# Optimized Trading System

Hệ thống trading đã được tối ưu hóa với khả năng optimization tham số chiến lược và backtest với tham số tối ưu.

## Cấu Trúc Mới

### 1. Trading Engine (`src/core/trading_engine.py`)
- **Bỏ phần backtest cũ**: Đã loại bỏ logic backtest cũ trong trading engine
- **Tích hợp BacktestEngine**: Sử dụng `BacktestEngine` để chạy backtest
- **Thêm optimization**: Tích hợp `StrategyOptimizer` để tối ưu hóa tham số

### 2. Backtest Engine (`src/backtesting/backtest_engine.py`)
- **Chạy nhiều chiến lược**: Hỗ trợ chạy 1 hoặc nhiều chiến lược cùng lúc
- **Standardized OHLCV**: Sử dụng format dữ liệu chuẩn
- **Performance metrics**: Tính toán các metric hiệu suất chi tiết

### 3. Strategy Optimizer (`src/optimization/optimizer.py`)
- **Parameter optimization**: Tối ưu hóa tham số cho từng chiến lược
- **Strategy selection**: Lựa chọn chiến lược tối ưu
- **Portfolio creation**: Tạo portfolio từ các chiến lược tốt nhất

## Các Tính Năng Chính

### 1. Strategy Optimization
```python
from src.core.trading_engine import TradingEngine

# Initialize engine
engine = TradingEngine()

# Optimize strategies
results = engine.optimize_strategies(
    strategy_types=['sma_crossover', 'rsi', 'macd'],
    start_date="2024-01-01",
    end_date="2024-05-31",
    optimization_metric='sharpe_ratio',
    max_combinations_per_strategy=50
)
```

### 2. Optimized Backtest
```python
# Run backtest with optimized strategies
results = engine.run_optimized_backtest(
    start_date="2024-01-01",
    end_date="2024-05-31",
    strategy_types=['sma_crossover', 'rsi']
)
```

### 3. Strategy Selection
```python
from src.optimization.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(config)

# Select optimal strategy
optimal_strategy = optimizer.select_optimal_strategy(
    strategy_types=['sma_crossover', 'rsi', 'macd'],
    metric='sharpe_ratio'
)

# Get strategy ranking
rankings = optimizer.get_strategy_ranking(
    strategy_types=['sma_crossover', 'rsi', 'macd'],
    metric='sharpe_ratio'
)

# Create optimal portfolio
portfolio = optimizer.create_optimal_strategy_portfolio(
    strategy_types=['sma_crossover', 'rsi', 'macd'],
    top_n=3,
    metric='sharpe_ratio'
)
```

## Workflow Hoàn Chỉnh

### Phase 1: Optimization
1. **Parameter Grid Generation**: Tạo các combination tham số
2. **Strategy Optimization**: Tối ưu hóa tham số cho từng chiến lược
3. **Best Parameters Selection**: Chọn tham số tốt nhất

### Phase 2: Strategy Selection
1. **Performance Comparison**: So sánh hiệu suất các chiến lược
2. **Optimal Strategy Selection**: Chọn chiến lược tối ưu
3. **Portfolio Creation**: Tạo portfolio từ top strategies

### Phase 3: Backtest
1. **Optimized Backtest**: Chạy backtest với tham số đã tối ưu
2. **Performance Analysis**: Phân tích hiệu suất
3. **Results Storage**: Lưu kết quả

## Các Method Chính

### TradingEngine Methods
- `optimize_strategies()`: Tối ưu hóa tham số cho nhiều chiến lược
- `get_optimized_strategies()`: Lấy các chiến lược đã tối ưu
- `run_optimized_backtest()`: Chạy backtest với chiến lược tối ưu
- `get_optimization_summary()`: Lấy tóm tắt kết quả optimization
- `save_optimization_results()`: Lưu kết quả optimization
- `load_optimization_results()`: Tải kết quả optimization

### StrategyOptimizer Methods
- `select_optimal_strategy()`: Chọn chiến lược tối ưu
- `get_strategy_ranking()`: Xếp hạng các chiến lược
- `create_optimal_strategy_portfolio()`: Tạo portfolio tối ưu
- `create_optimized_strategy()`: Tạo chiến lược với tham số tối ưu

## Các Metric Được Sử Dụng

### Performance Metrics
- **Total Return**: Tổng lợi nhuận
- **Sharpe Ratio**: Tỷ lệ Sharpe
- **Max Drawdown**: Drawdown tối đa
- **Win Rate**: Tỷ lệ thắng
- **Profit Factor**: Hệ số lợi nhuận
- **Volatility**: Độ biến động

### Optimization Metrics
- **sharpe_ratio**: Tối ưu theo Sharpe ratio
- **total_return**: Tối ưu theo tổng lợi nhuận
- **profit_factor**: Tối ưu theo profit factor

## Cách Sử Dụng

### 1. Chạy Demo
```bash
python demo_optimized_trading.py
```

### 2. Chạy Optimization Cơ Bản
```python
from src.core.trading_engine import TradingEngine

engine = TradingEngine()
results = engine.optimize_strategies(
    strategy_types=['sma_crossover', 'rsi'],
    start_date="2024-01-01",
    end_date="2024-05-31"
)
```

### 3. Chạy Optimized Backtest
```python
results = engine.run_optimized_backtest(
    start_date="2024-01-01",
    end_date="2024-05-31",
    strategy_types=['sma_crossover', 'rsi']
)
```

### 4. Strategy Selection
```python
from src.optimization.optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(config)
optimal_strategy = optimizer.select_optimal_strategy(
    strategy_types=['sma_crossover', 'rsi', 'macd'],
    metric='sharpe_ratio'
)
```

## Các File Output

### Optimization Results
- `data/optimization_results.json`: Kết quả optimization
- `data/demo_optimization_results.json`: Kết quả optimization demo

### Backtest Results
- `data/backtest_results.json`: Kết quả backtest
- `data/demo_backtest_results.json`: Kết quả backtest demo

### Reports
- Optimization summary: Tóm tắt kết quả optimization
- Strategy rankings: Xếp hạng các chiến lược
- Portfolio composition: Thành phần portfolio

## Lưu Ý Quan Trọng

1. **Data Requirements**: Đảm bảo có dữ liệu OHLCV trong khoảng thời gian được chỉ định
2. **Computational Resources**: Optimization có thể mất nhiều thời gian với nhiều combination
3. **Parameter Limits**: Điều chỉnh `max_combinations_per_strategy` để cân bằng giữa độ chính xác và thời gian
4. **Strategy Selection**: Luôn xem xét các metric rủi ro cùng với lợi nhuận

## Troubleshooting

### Lỗi Thường Gặp

1. **No data available**: Kiểm tra lại khoảng thời gian và dữ liệu
2. **Memory issues**: Giảm `max_combinations_per_strategy`
3. **Import errors**: Đảm bảo tất cả dependencies đã được cài đặt
4. **Circular imports**: Sử dụng lazy imports trong trading engine

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Tương Lai Phát Triển

1. **Machine Learning Optimization**: Sử dụng ML để tối ưu hóa tham số
2. **Multi-Objective Optimization**: Tối ưu hóa nhiều metric cùng lúc
3. **Real-time Optimization**: Tối ưu hóa real-time
4. **Portfolio Optimization**: Tối ưu hóa portfolio thay vì từng chiến lược riêng lẻ
5. **Risk-Adjusted Selection**: Lựa chọn chiến lược dựa trên risk-adjusted metrics 