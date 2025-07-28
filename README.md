# Auto Trading System

A comprehensive automated trading system built with Python, featuring real-time market data processing, multiple trading strategies, risk management, and a web-based dashboard.

## Features

- **Real-time Market Data**: Live price feeds and historical data management
- **Multiple Trading Strategies**: SMA crossover, RSI, MACD, and custom strategies
- **Technical Indicators**: Modular indicator library (SMA, RSI, MACD, Bollinger Bands)
- **Advanced Charting**: Lightweight-charts based OHLC visualization with volume and signals
- **Risk Management**: Position sizing, stop-loss, and portfolio limits
- **Backtesting Engine**: Historical strategy performance analysis
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

3. **Run Backtesting with Different Strategies**:
   ```bash
   # SMA Crossover Strategy
   python src/main.py --mode backtest --strategy sma_crossover
   
   # RSI Strategy
   python src/main.py --mode backtest --strategy rsi
   
   # MACD Strategy
   python src/main.py --mode backtest --strategy macd
   ```

4. **Start Live Trading**:
   ```bash
   python src/main.py --mode live --strategy sma_crossover
   python src/main.py --mode live --strategy rsi
   python src/main.py --mode live --strategy macd
   ```

5. **Launch Dashboard**:
   ```bash
   python run_dashboard.py
   # Or directly with Streamlit:
   streamlit run dashboard/streamlit_app.py
   ```

6. **Launch Trading Chart**:
   ```bash
   python run_trading_chart.py
   # Opens interactive charts with indicators and signals
   ```

## Configuration

Edit `config/config.yaml` to customize:
- Trading parameters
- Risk management settings
- Data sources
- Strategy parameters

## Chart Features

The trading chart dashboard provides:

- **📊 OHLC Candlestick Charts**: Professional candlestick visualization
- **📈 Volume Display**: Volume bars synchronized with price charts
- **🎯 Trading Signals**: Buy/sell markers with strategy integration
- **📊 Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Real-time Data**: Live data from Yahoo Finance API

See `dashboard/TRADING_CHART_README.md` for detailed documentation.

## Risk Warning

This system is for educational purposes. Always test thoroughly before using with real money. Trading involves substantial risk of loss.

## License

MIT License
