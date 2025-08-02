# Auto Trading System

A comprehensive automated trading system built with Python, featuring real-time market data processing, multiple trading strategies, risk management, strategy optimization, and a web-based dashboard.

## Features

- **Real-time Market Data**: Live price feeds and historical data management
- **Multiple Trading Strategies**: SMA crossover, RSI, MACD, and custom strategies
- **Technical Indicators**: Modular indicator library (SMA, RSI, MACD, Bollinger Bands)
- **Advanced Charting**: Lightweight-charts based OHLC visualization with volume and signals
- **Risk Management**: Position sizing, stop-loss, and portfolio limits with comprehensive metrics
- **Backtesting Engine**: Historical strategy performance analysis with risk management integration
- **Strategy Optimization**: Parameter optimization using risk management metrics
- **Web Dashboard**: Modern Streamlit-based monitoring and control interface
- **Trading Chart Dashboard**: Interactive charts with indicators and signals
- **Configuration Management**: Flexible strategy and system parameters
- **Logging & Monitoring**: Comprehensive logging and performance tracking

## Project Structure

```
autotrading_system/
├── src/
│   ├── core/           # Core trading engine
│   ├── strategies/     # Trading strategies
│   ├── indicators/     # Technical indicators
│   ├── data/          # Data management
│   ├── risk/          # Risk management
│   ├── backtesting/   # Backtesting framework
│   ├── optimization/  # Strategy parameter optimization
│   └── utils/         # Utilities and helpers
├── config/            # Configuration files
├── tests/             # Unit tests
├── dashboard/         # Web dashboard
├── logs/              # Log files
└── data/              # Market data storage
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**:
   ```bash
   # Run setup to create config file
   python setup.py
   # Edit config/config.yaml with your settings
   ```

3. **Run Backtesting**:
   ```bash
   # Test backtesting functionality
   python test_backtest.py
   
   # Run full backtest with all strategies
   python run_backtest.py
   ```

4. **Optimize Strategy Parameters**:
   ```bash
   # Test optimization functionality
   python test_optimization.py
   
   # Run full optimization for all strategies
   python run_optimization.py
   ```

5. **Run Backtesting with Different Strategies**:
   ```bash
   # SMA Crossover Strategy
   python src/main.py --mode backtest --strategy sma_crossover
   
   # RSI Strategy
   python src/main.py --mode backtest --strategy rsi
   
   # MACD Strategy
   python src/main.py --mode backtest --strategy macd
   ```

6. **Start Live Trading**:
   ```bash
   python src/main.py --mode live --strategy sma_crossover
   python src/main.py --mode live --strategy rsi
   python src/main.py --mode live --strategy macd
   ```

7. **Launch Dashboard**:
   ```bash
   python run_dashboard.py
   # Or directly with Streamlit:
   streamlit run dashboard/streamlit_app.py
   ```

8. **Launch Trading Chart**:
   ```bash
   python run_trading_chart.py
   # Opens interactive charts with indicators and signals
   ```

## Backtesting & Optimization

### Backtesting Engine
The system includes a comprehensive backtesting engine that integrates with risk management:

- **Multi-strategy backtesting**: Test multiple strategies simultaneously
- **Risk management integration**: Uses built-in risk metrics from `risk_manager.py`
- **Performance metrics**: Sharpe ratio, drawdown, win rate, profit factor
- **Trade tracking**: Complete trade history and analysis
- **Visualization**: Portfolio charts and performance plots

### Strategy Optimization
The optimization module finds optimal parameters for each strategy:

- **Parameter optimization**: Systematic testing of parameter combinations
- **Risk-based optimization**: Uses risk management metrics as optimization criteria
- **Multiple optimization metrics**: Sharpe ratio, total return, profit factor, etc.
- **Parallel processing**: Efficient multi-core execution for large parameter grids
- **Results persistence**: Save and load optimization results

### Usage Examples

```python
# Run backtesting
from src.backtesting import BacktestEngine
backtest_engine = BacktestEngine(config)
results = backtest_engine.run_backtest(strategies, start_date, end_date)

# Optimize strategy parameters
from src.optimization import StrategyOptimizer
optimizer = StrategyOptimizer(config)
result = optimizer.optimize_strategy(
    strategy_type='sma_crossover',
    start_date='2024-01-01',
    end_date='2024-05-31',
    optimization_metric='sharpe_ratio'
)
```

## Configuration

Edit `config/config.yaml` to customize:
- Trading parameters
- Risk management settings
- Data sources
- Strategy parameters
- Optimization settings

## Chart Features

The trading chart dashboard provides:

- **📊 OHLC Candlestick Charts**: Professional candlestick visualization
- **📈 Volume Display**: Volume bars synchronized with price charts
- **🎯 Trading Signals**: Buy/sell markers with strategy integration
- **📊 Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Real-time Data**: Live data from Yahoo Finance API

See `dashboard/TRADING_CHART_README.md` for detailed documentation.

## Risk Management Integration

The system includes comprehensive risk management:

- **Position sizing**: Based on risk parameters and portfolio limits
- **Stop loss/take profit**: Automatic execution based on risk settings
- **Drawdown monitoring**: Real-time tracking with maximum limits
- **Portfolio risk limits**: Maximum portfolio risk enforcement
- **Risk-adjusted returns**: Sharpe ratio and other risk metrics

## Performance Metrics

The system calculates comprehensive performance metrics:

- **Return metrics**: Total return, annualized return
- **Risk metrics**: Sharpe ratio, volatility, maximum drawdown
- **Trade metrics**: Win rate, profit factor, average trade
- **Risk management metrics**: Current drawdown, portfolio volatility

## Risk Warning

This system is for educational purposes. Always test thoroughly before using with real money. Trading involves substantial risk of loss.

## License

MIT License
