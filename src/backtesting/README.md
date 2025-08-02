# Backtesting Module

The backtesting module provides a comprehensive framework for testing trading strategies and analyzing their performance using built-in risk management metrics.

## Features

- **Multi-strategy backtesting**: Test multiple strategies simultaneously
- **Risk management integration**: Uses built-in risk metrics from `risk_manager.py`
- **Performance metrics**: Comprehensive performance analysis including:
  - Total and annualized returns
  - Sharpe ratio
  - Maximum drawdown
  - Win rate and profit factor
  - Volatility analysis
- **Visualization**: Generate plots and charts of backtest results
- **Report generation**: Detailed performance reports
- **Trade tracking**: Complete trade history and analysis

## Usage

### Basic Usage

```python
from src.backtesting import BacktestEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.utils.config_manager import ConfigManager

# Load configuration
config = ConfigManager("config/config.yaml")

# Initialize backtest engine
backtest_engine = BacktestEngine(config)

# Create strategies
strategies = {
    'SMA_Crossover': SMACrossoverStrategy({
        'short_window': 20,
        'long_window': 50
    })
}

# Run backtest
results = backtest_engine.run_backtest(
    strategies, 
    start_date="2024-01-01", 
    end_date="2024-05-31"
)

# Generate report
report = backtest_engine.generate_report()
print(report)

# Plot results
backtest_engine.plot_results(save_path="backtest_results.png")
```

### Running Multiple Strategies

```python
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy

strategies = {
    'SMA_Crossover': SMACrossoverStrategy({
        'short_window': 20,
        'long_window': 50
    }),
    'RSI': RSIStrategy({
        'period': 14,
        'oversold': 30,
        'overbought': 70
    }),
    'MACD': MACDStrategy({
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    })
}

results = backtest_engine.run_backtest(strategies, start_date, end_date)
```

## Performance Metrics

The backtesting engine calculates the following metrics:

### Return Metrics
- **Total Return**: Overall percentage return
- **Annualized Return**: Annualized return rate
- **Sharpe Ratio**: Risk-adjusted return measure

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Current Drawdown**: Current drawdown level

### Trade Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Total Trades**: Number of completed trades
- **Average Trade**: Average profit/loss per trade

## Risk Management Integration

The backtesting engine integrates with the risk manager to provide:

- **Position sizing**: Based on risk parameters
- **Stop loss**: Automatic stop loss execution
- **Take profit**: Automatic take profit execution
- **Portfolio risk limits**: Maximum portfolio risk enforcement
- **Drawdown monitoring**: Real-time drawdown tracking

## Configuration

The backtesting engine uses the following configuration parameters:

```yaml
# Trading Settings
trading:
  initial_capital: 100000
  commission: 0.001  # 0.1% commission per trade
  symbols: ["VCB"]

# Risk Management
risk:
  max_position_size: 0.2  # 20% of portfolio per position
  max_portfolio_risk: 0.05  # 5% max portfolio risk
  stop_loss: 0.05  # 5% stop loss
  take_profit: 0.15  # 15% take profit
  max_drawdown: 0.20  # 20% max drawdown

# Data Settings
data:
  start_date: "2024-01-01"
  end_date: "2024-05-31"
  source: "vnstock"
```

## Output Files

The backtesting engine can generate:

1. **Performance Report**: Text-based performance summary
2. **Results JSON**: Detailed results in JSON format
3. **Plots**: Visual charts of portfolio performance
4. **Trade Log**: Complete trade history

## Example Output

```
============================================================
BACKTEST REPORT
============================================================

Strategy: SMA_Crossover
Final Portfolio Value: $105,234.56
Total Return: 5.23%
Annualized Return: 12.45%
Volatility: 15.67%
Sharpe Ratio: 0.78
Max Drawdown: -8.34%
Win Rate: 52.3%
Profit Factor: 1.45
Total Trades: 24
Average Trade: $234.56
Current Drawdown: -2.1%
Portfolio Volatility: 12.34%
```

## Running Backtests

### Using the provided script:

```bash
python run_backtest.py
```

### Using the test script:

```bash
python test_backtest.py
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Plotting
- seaborn: Statistical visualization
- loguru: Logging

## Integration with Existing System

The backtesting module integrates seamlessly with:

- **Data Manager**: Historical data retrieval
- **Risk Manager**: Risk metrics and controls
- **Strategies**: All implemented trading strategies
- **Configuration**: Centralized configuration management

This provides a complete backtesting solution that leverages the existing risk management infrastructure while providing comprehensive performance analysis. 