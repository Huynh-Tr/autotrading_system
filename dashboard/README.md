# Auto Trading System - Streamlit Dashboard

A modern, interactive dashboard for monitoring and controlling the auto trading system.

## Features

- **📊 Real-time Portfolio Overview**: Live portfolio value, cash, and P&L tracking
- **📈 Performance Analytics**: Sharpe ratio, drawdown, volatility, and win rate analysis
- **💼 Position Management**: Current positions with P&L tracking
- **📋 Trade History**: Complete trade history with filtering
- **🎛️ System Controls**: Start/stop trading, strategy selection, and mode switching
- **📱 Responsive Design**: Works on desktop and mobile devices

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install streamlit plotly pandas numpy
   ```

2. **Launch Dashboard**:
   ```bash
   python run_dashboard.py
   # Or directly:
   streamlit run dashboard/streamlit_app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:8501`

## Dashboard Sections

### 📊 Overview Tab
- Portfolio value and performance metrics
- Interactive portfolio performance chart
- Risk metrics with gauge visualizations
- Key performance indicators

### 📈 Performance Tab
- Returns distribution analysis
- Cumulative returns chart
- Drawdown analysis
- Performance comparison tools

### 💼 Positions Tab
- Current open positions
- Position allocation pie chart
- P&L tracking per position
- Position details and metrics

### 📋 Trades Tab
- Complete trade history
- Trade side distribution
- Volume analysis by symbol
- Trade performance metrics

## Sidebar Controls

- **System Status**: Running/Stopped/Paused
- **Trading Mode**: Backtest/Paper Trading/Live Trading
- **Strategy Selection**: Choose active trading strategy
- **Quick Actions**: Refresh data, run backtests
- **System Info**: Version, uptime, last update

## Configuration

The dashboard automatically loads configuration from `config/config.yaml`. Key settings:

```yaml
dashboard:
  host: "localhost"
  port: 8501
  debug: false
```

## Customization

### Adding New Metrics
1. Modify the mock data functions in `streamlit_app.py`
2. Add new visualizations using Plotly
3. Update the sidebar controls as needed

### Connecting Real Data
1. Replace mock data functions with real API calls
2. Integrate with the trading engine for live data
3. Add real-time updates using Streamlit's caching

### Styling
- Custom CSS is included in the app
- Modify the `st.markdown` section for custom styling
- Use Streamlit's built-in theming options

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   streamlit run dashboard/streamlit_app.py --server.port 8502
   ```

2. **Import errors**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration not found**:
   ```bash
   cp config/config.example.yaml config/config.yaml
   ```

### Performance Tips

- Use `@st.cache_data` for expensive computations
- Limit data refresh frequency
- Optimize chart rendering with Plotly
- Use pagination for large datasets

## Development

### File Structure
```
dashboard/
├── streamlit_app.py      # Main dashboard application
├── README.md            # This file
└── templates/           # Legacy Flask templates (not used)
```

### Adding New Features

1. **New Tab**: Add to the `st.tabs()` call
2. **New Chart**: Use Plotly or Streamlit's built-in charts
3. **New Control**: Add to the sidebar section
4. **New Data Source**: Modify the data loading functions

### Testing

Run the Streamlit test suite:
```bash
python test_streamlit.py
```

## Security Notes

- Dashboard runs on localhost by default
- No authentication implemented (add for production)
- Sensitive data should be encrypted
- Use HTTPS for production deployments

## License

MIT License - see main project README for details. 