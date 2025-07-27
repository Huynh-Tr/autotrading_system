# Trading Strategies Documentation

This document provides detailed information about all trading strategies available in the Auto Trading System.

## Available Strategies

### 1. **SMA Crossover Strategy** (`sma_crossover`)

**Description**: Simple Moving Average crossover strategy that generates buy/sell signals based on short and long-term moving average crossovers.

**Parameters**:
- `short_window` (default: 20): Short-term SMA period
- `long_window` (default: 50): Long-term SMA period

**Signal Logic**:
- **Buy Signal**: When short SMA crosses above long SMA (Golden Cross)
- **Sell Signal**: When short SMA crosses below long SMA (Death Cross)
- **Hold**: When no crossover occurs

**Usage**:
```bash
python src/main.py --mode backtest --strategy sma_crossover
```

**Configuration**:
```yaml
strategies:
  sma_crossover:
    enabled: true
    short_window: 20
    long_window: 50
```

### 2. **RSI Strategy** (`rsi`)

**Description**: Relative Strength Index strategy that identifies overbought and oversold conditions to generate trading signals.

**Parameters**:
- `period` (default: 14): RSI calculation period
- `overbought_threshold` (default: 70): RSI level considered overbought
- `oversold_threshold` (default: 30): RSI level considered oversold
- `confirmation_period` (default: 2): Number of periods for confirmation

**Signal Logic**:
- **Buy Signal**: 
  - RSI crosses below oversold threshold (30)
  - Sustained oversold condition for confirmation period
- **Sell Signal**: 
  - RSI crosses above overbought threshold (70)
  - Sustained overbought condition for confirmation period
- **Hold**: When RSI is between thresholds

**Usage**:
```bash
python src/main.py --mode backtest --strategy rsi
```

**Configuration**:
```yaml
strategies:
  rsi:
    enabled: true
    period: 14
    overbought_threshold: 70
    oversold_threshold: 30
    confirmation_period: 2
```

### 3. **MACD Strategy** (`macd`)

**Description**: Moving Average Convergence Divergence strategy that uses MACD line, signal line, and histogram for trading decisions.

**Parameters**:
- `fast_period` (default: 12): Fast EMA period
- `slow_period` (default: 26): Slow EMA period
- `signal_period` (default: 9): Signal line EMA period
- `histogram_threshold` (default: 0.0): Minimum histogram value for signals
- `confirmation_period` (default: 1): Number of periods for confirmation

**Signal Logic**:
- **Buy Signal**:
  - MACD line crosses above signal line (bullish crossover)
  - Histogram turns positive with MACD above signal
  - MACD crosses above zero line
  - Sustained bullish trend
- **Sell Signal**:
  - MACD line crosses below signal line (bearish crossover)
  - Histogram turns negative with MACD below signal
  - MACD crosses below zero line
  - Sustained bearish trend
- **Hold**: When no clear signal conditions are met

**Usage**:
```bash
python src/main.py --mode backtest --strategy macd
```

**Configuration**:
```yaml
strategies:
  macd:
    enabled: true
    fast_period: 12
    slow_period: 26
    signal_period: 9
    histogram_threshold: 0.0
    confirmation_period: 1
```

## Strategy Comparison

| Strategy | Type | Best For | Risk Level | Signal Frequency |
|----------|------|----------|------------|------------------|
| **SMA Crossover** | Trend Following | Trending markets | Medium | Low |
| **RSI** | Mean Reversion | Sideways markets | Medium | Medium |
| **MACD** | Trend Following | Trending markets | Medium | Medium |

## Strategy Selection Guide

### **Choose SMA Crossover when**:
- Markets are trending strongly
- You want fewer, higher-quality signals
- You prefer simple, well-established indicators
- You want to catch major trend changes

### **Choose RSI when**:
- Markets are ranging/sideways
- You want to identify overbought/oversold conditions
- You prefer mean reversion strategies
- You want more frequent trading opportunities

### **Choose MACD when**:
- Markets are trending
- You want to identify momentum changes
- You prefer multiple confirmation signals
- You want to catch both trend and momentum shifts

## Performance Considerations

### **SMA Crossover**:
- **Pros**: Simple, reliable, catches major trends
- **Cons**: Lagging indicator, may miss quick reversals
- **Best Timeframe**: Daily or longer

### **RSI**:
- **Pros**: Identifies extremes, good for range-bound markets
- **Cons**: Can stay overbought/oversold for extended periods
- **Best Timeframe**: 4-hour to daily

### **MACD**:
- **Pros**: Multiple signal types, good trend confirmation
- **Cons**: Can generate false signals in choppy markets
- **Best Timeframe**: 1-hour to daily

## Risk Management Integration

All strategies integrate with the system's risk management:

1. **Position Sizing**: Based on portfolio risk limits
2. **Stop Loss**: Automatic stop-loss orders
3. **Take Profit**: Automatic take-profit orders
4. **Portfolio Limits**: Maximum position and portfolio risk limits
5. **Drawdown Protection**: Automatic trading halt on excessive drawdown

## Testing Strategies

### **Individual Strategy Testing**:
```bash
# Test RSI strategy
python test_strategies.py

# Test specific strategy
python src/main.py --mode backtest --strategy rsi
```

### **Strategy Comparison**:
```bash
# Test all strategies on same data
python src/main.py --mode backtest --strategy sma_crossover
python src/main.py --mode backtest --strategy rsi
python src/main.py --mode backtest --strategy macd
```

## Custom Strategy Development

To add a new strategy:

1. **Create Strategy Class**:
   ```python
   from src.strategies.base_strategy import BaseStrategy
   
   class MyStrategy(BaseStrategy):
       def __init__(self, config):
           super().__init__("my_strategy", config)
           # Initialize parameters
       
       def generate_signals(self, data):
           # Implement signal generation logic
           return signals
   ```

2. **Add to Main System**:
   ```python
   # In src/main.py
   from src.strategies.my_strategy import MyStrategy
   
   if strategy_name == "my_strategy":
       strategy_config = config.get("strategies.my_strategy", {})
       strategy = MyStrategy(strategy_config)
       engine.add_strategy(strategy)
   ```

3. **Update Configuration**:
   ```yaml
   strategies:
     my_strategy:
       enabled: true
       param1: value1
       param2: value2
   ```

## Best Practices

1. **Always backtest** before live trading
2. **Use multiple timeframes** for confirmation
3. **Combine strategies** for better results
4. **Monitor performance** regularly
5. **Adjust parameters** based on market conditions
6. **Use proper risk management** with all strategies

## Performance Metrics

Each strategy provides:
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Typical holding period
- **Volatility**: Strategy risk level

## Configuration Examples

### **Conservative RSI Settings**:
```yaml
strategies:
  rsi:
    period: 20
    overbought_threshold: 75
    oversold_threshold: 25
    confirmation_period: 3
```

### **Aggressive MACD Settings**:
```yaml
strategies:
  macd:
    fast_period: 8
    slow_period: 21
    signal_period: 5
    histogram_threshold: 0.5
    confirmation_period: 1
```

### **Balanced SMA Settings**:
```yaml
strategies:
  sma_crossover:
    short_window: 15
    long_window: 40
```

This comprehensive strategy documentation helps users understand, configure, and effectively use all available trading strategies in the Auto Trading System. 