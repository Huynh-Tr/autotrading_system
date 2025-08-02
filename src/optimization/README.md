# Strategy Optimization Module

The optimization module provides comprehensive parameter optimization for trading strategies using risk management metrics from `risk_manager.py`. It systematically tests different parameter combinations to find the optimal settings for each strategy.

## Features

- **Multi-strategy optimization**: Optimize parameters for SMA, RSI, MACD, and other strategies
- **Risk-based optimization**: Uses risk management metrics for optimization criteria
- **Parallel processing**: Efficient parallel execution for large parameter grids
- **Multiple optimization metrics**: Optimize by Sharpe ratio, total return, profit factor, etc.
- **Parameter filtering**: Filter parameter combinations to focus on specific ranges
- **Comprehensive reporting**: Detailed optimization reports and visualizations
- **Results persistence**: Save and load optimization results

## Core Components

### ParameterGrid
Defines parameter ranges for different strategy types and generates all possible combinations.

### StrategyOptimizer
Main optimization engine that tests parameter combinations using the backtesting engine.

## Usage

### Basic Optimization

```python
from src.optimization import StrategyOptimizer
from src.utils.config_manager import ConfigManager

# Load configuration
config = ConfigManager("config/config.yaml")

# Initialize optimizer
optimizer = StrategyOptimizer(config)

# Optimize a single strategy
result = optimizer.optimize_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-05-31',
    optimization_metric='sharpe_ratio',
    max_combinations=50
)

# Get best parameters
best_params = result['best_parameters']['parameters']
print(f"Best parameters: {best_params}")
```

### Multi-Strategy Optimization

```python
# Optimize multiple strategies
strategy_types = ['sma_crossover', 'rsi', 'macd']
results = optimizer.optimize_multiple_strategies(
    strategy_types=strategy_types,
    start_date='2024-01-01',
    end_date='2024-05-31',
    optimization_metric='sharpe_ratio',
    max_combinations_per_strategy=30
)
```

### Parameter Filtering

```python
# Filter parameter combinations
filters = {
    'short_window': [10, 20, 30],
    'long_window': [50, 60, 70]
}

result = optimizer.optimize_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-05-31',
    filters=filters
)
```

## Optimization Metrics

The optimizer can optimize based on different metrics:

- **sharpe_ratio**: Risk-adjusted returns (default)
- **total_return**: Total percentage return
- **profit_factor**: Ratio of gross profit to gross loss
- **win_rate**: Percentage of profitable trades
- **max_drawdown**: Maximum drawdown (minimize)

## Parameter Grids

### SMA Crossover Strategy
```python
{
    'short_window': [5, 10, 15, 20, 25, 30],
    'long_window': [30, 40, 50, 60, 70, 80, 90, 100]
}
```

### RSI Strategy
```python
{
    'period': [10, 14, 20, 30],
    'oversold': [20, 25, 30, 35],
    'overbought': [65, 70, 75, 80]
}
```

### MACD Strategy
```python
{
    'fast_period': [8, 10, 12, 15, 20],
    'slow_period': [20, 26, 30, 35, 40],
    'signal_period': [7, 9, 12, 15]
}
```

## Advanced Usage

### Custom Parameter Grids

```python
from src.optimization import ParameterGrid

grid = ParameterGrid()

# Add custom parameter grid
custom_params = {
    'custom_strategy': {
        'param1': [1, 2, 3],
        'param2': [10, 20, 30]
    }
}
grid.add_custom_grid('custom_strategy', custom_params['custom_strategy'])

# Get combinations
combinations = grid.get_parameter_combinations('custom_strategy')
```

### Parallel vs Sequential Processing

```python
# Use parallel processing for large grids
result = optimizer.optimize_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-05-31',
    use_parallel=True  # Default for large grids
)

# Use sequential processing for small grids or debugging
result = optimizer.optimize_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-05-31',
    use_parallel=False
)
```

### Creating Optimized Strategies

```python
# Create strategy with best parameters
optimized_strategy = optimizer.create_optimized_strategy('sma_crossover')

# Create strategy with custom parameters
custom_params = {'short_window': 15, 'long_window': 45}
custom_strategy = optimizer.create_optimized_strategy(
    'sma_crossover', 
    use_best_parameters=False,
    custom_parameters=custom_params
)
```

## Results Analysis

### Getting Top Parameters

```python
# Get top 5 parameter combinations
top_params = optimizer.get_top_parameters('sma_crossover', top_n=5)

for i, param_set in enumerate(top_params, 1):
    params = param_set['parameters']
    metrics = param_set['metrics']
    print(f"Rank {i}: {params} -> Sharpe: {metrics['sharpe_ratio']:.3f}")
```

### Generating Reports

```python
# Generate comprehensive report
report = optimizer.generate_optimization_report()
print(report)

# Generate report for specific strategy
report = optimizer.generate_optimization_report('sma_crossover')
print(report)
```

### Visualization

```python
# Plot optimization results
optimizer.plot_optimization_results(
    'sma_crossover',
    save_path='optimization_results.png'
)
```

## Persistence

### Saving Results

```python
# Save optimization results
optimizer.save_optimization_results('data/optimization_results.json')
```

### Loading Results

```python
# Load previously saved results
optimizer.load_optimization_results('data/optimization_results.json')

# Use loaded best parameters
optimized_strategy = optimizer.create_optimized_strategy('sma_crossover')
```

## Running Optimization

### Using the provided script:

```bash
python run_optimization.py
```

### Using the test script:

```bash
python test_optimization.py
```

### Custom optimization:

```python
from run_optimization import optimize_single_strategy

# Optimize single strategy with custom settings
result = optimize_single_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-03-31',
    optimization_metric='total_return',
    max_combinations=20
)
```

## Example Output

```
============================================================
STRATEGY OPTIMIZATION REPORT
============================================================

Strategy: SMA_CROSSOVER
Optimization Metric: sharpe_ratio
Total Combinations Tested: 48
Best Parameters: {'short_window': 15, 'long_window': 60}
Best Performance Metrics:
  Total Return: 12.45%
  Annualized Return: 28.67%
  Sharpe Ratio: 1.23
  Max Drawdown: -5.67%
  Win Rate: 58.3%
  Profit Factor: 1.89
  Total Trades: 24
```

## Configuration

The optimization module uses the same configuration as the backtesting engine:

```yaml
# Optimization settings
optimization:
  default_metric: 'sharpe_ratio'
  max_combinations_per_strategy: 50
  use_parallel: true
  save_results: true
  plot_results: true
```

## Integration with Risk Management

The optimization module leverages all risk management metrics:

- **Position sizing**: Based on risk parameters
- **Stop loss/take profit**: Automatic execution
- **Drawdown monitoring**: Real-time tracking
- **Portfolio risk limits**: Maximum risk enforcement
- **Risk-adjusted returns**: Sharpe ratio calculation

## Performance Considerations

- **Parallel processing**: Automatically used for grids > 10 combinations
- **Memory management**: Results are stored efficiently
- **Progress tracking**: Real-time progress updates
- **Error handling**: Robust error handling for failed combinations

## Best Practices

1. **Start small**: Test with limited combinations first
2. **Use appropriate metrics**: Choose optimization metric based on goals
3. **Filter parameters**: Use filters to focus on relevant ranges
4. **Validate results**: Always backtest optimized strategies
5. **Save results**: Persist optimization results for future use
6. **Monitor performance**: Track optimization progress and performance

This optimization module provides a comprehensive solution for finding optimal strategy parameters while maintaining integration with the existing risk management infrastructure. 