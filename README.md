# Auto Trading System

A comprehensive automated trading system built with Python, featuring real-time market data processing, multiple trading strategies, risk management, and a web-based dashboard.

## Features

- **Real-time Market Data**: Live price feeds and historical data management
- **Multiple Trading Strategies**: SMA crossover, RSI, MACD, and custom strategies
- **Risk Management**: Position sizing, stop-loss, and portfolio limits
- **Backtesting Engine**: Historical strategy performance analysis
- **Web Dashboard**: Modern Streamlit-based monitoring and control interface
- **Configuration Management**: Flexible strategy and system parameters
- **Logging & Monitoring**: Comprehensive logging and performance tracking

## Project Structure

```
autotrading_system/
├── src/
│   ├── core/           # Core trading engine
│   ├── strategies/     # Trading strategies
│   ├── data/          # Data management
│   ├── risk/          # Risk management
│   ├── backtesting/   # Backtesting framework
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
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your settings
   ```

3. **Run Backtesting**:
   ```bash
   python src/main.py --mode backtest --strategy sma_crossover
   ```

4. **Start Live Trading**:
   ```bash
   python src/main.py --mode live --strategy sma_crossover
   ```

5. **Launch Dashboard**:
   ```bash
   python run_dashboard.py
   # Or directly with Streamlit:
   streamlit run dashboard/streamlit_app.py
   ```

## Configuration

Edit `config/config.yaml` to customize:
- Trading parameters
- Risk management settings
- Data sources
- Strategy parameters

## Risk Warning

This system is for educational purposes. Always test thoroughly before using with real money. Trading involves substantial risk of loss.

## License

MIT License
