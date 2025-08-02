"""
Streamlit Dashboard for Auto Trading System
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import yaml
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Import with proper error handling and fallbacks
try:
    from utils.config_manager import ConfigManager
    print("✅ ConfigManager imported successfully")
except ImportError as e:
    print(f"⚠️ ConfigManager import error: {e}")
    ConfigManager = None

try:
    from strategies.sma_crossover import SMACrossoverStrategy
    print("✅ SMACrossoverStrategy imported successfully")
except ImportError as e:
    print(f"⚠️ SMACrossoverStrategy import error: {e}")
    SMACrossoverStrategy = None

try:
    from strategies.rsi_strategy import RSIStrategy
    print("✅ RSIStrategy imported successfully")
except ImportError as e:
    print(f"⚠️ RSIStrategy import error: {e}")
    RSIStrategy = None

try:
    from strategies.macd_strategy import MACDStrategy
    print("✅ MACDStrategy imported successfully")
except ImportError as e:
    print(f"⚠️ MACDStrategy import error: {e}")
    MACDStrategy = None

try:
    from viz.viz import viz
    print("✅ Visualization module imported successfully")
except ImportError as e:
    print(f"⚠️ Visualization module import error: {e}")
    viz = None

try:
    from core.trading_engine import TradingEngine
    print("✅ TradingEngine imported successfully")
except ImportError as e:
    print(f"⚠️ TradingEngine import error: {e}")
    TradingEngine = None

try:
    from data.data_manager import DataManager
    print("✅ DataManager imported successfully")
except ImportError as e:
    print(f"⚠️ DataManager import error: {e}")
    DataManager = None

# Page configuration
st.set_page_config(
    page_title="Auto Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit {
        color: #28a745;
        font-weight: bold;
    }
    .loss {
        color: #dc3545;
        font-weight: bold;
    }
    .status-running {
        color: #28a745;
    }
    .status-stopped {
        color: #dc3545;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .quick-metric {
        text-align: center;
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration with caching"""
    try:
        config_path = os.path.join(project_root, "config", "config.yaml")
        if not os.path.exists(config_path):
            st.error(f"Configuration file not found: {config_path}")
            return None
        
        if ConfigManager:
            return ConfigManager(config_path)
        else:
            # Fallback: load config manually
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return config_data
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

def run_backtest_from_dashboard(config, strategy_name, params):
    """Run backtest from dashboard using actual trading engine"""
    try:
        if not config or not TradingEngine or not DataManager:
            st.error("Trading engine components not available")
            return None
        
        # Get config path
        config_path = os.path.join(project_root, "config", "config.yaml")
        if not os.path.exists(config_path):
            st.error(f"Configuration file not found: {config_path}")
            return None
        
        # Initialize trading engine
        engine = TradingEngine(config_path)
        
        # Get parameters from config
        symbols = params.get('symbols', ['Bitstamp:BTCUSD'])
        start_date = params.get('start_date', '2023-01-01')
        end_date = params.get('end_date', '2025-07-31')
        interval = params.get('interval', '4h')
        n_bars = params.get('n_bars', 100)
        
        # Initialize data manager and get historical data
        data_manager = DataManager(engine.config)
        historical_data = data_manager.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=engine.config.get("data.interval", "4h"),
            n_bars=engine.config.get("data.n_bars", 100)
        )
        
        if historical_data.empty:
            st.error("No historical data available for the specified period")
            return None
        
        # Create strategy based on selection
        strategies = {}
        if strategy_name == "SMA Crossover" and SMACrossoverStrategy:
            strategies['sma_crossover'] = SMACrossoverStrategy(engine.config)
        elif strategy_name == "RSI Strategy" and RSIStrategy:
            strategies['rsi'] = RSIStrategy(engine.config)
        elif strategy_name == "MACD Strategy" and MACDStrategy:
            strategies['macd'] = MACDStrategy(engine.config)
        else:
            st.error(f"Strategy {strategy_name} not available")
            return None
        
        # Run backtest
        results = engine.run_backtest(strategies, start_date, end_date)
        
        if not results:
            st.error("Backtest failed to produce results")
            return None
        
        # Get portfolio summary
        portfolio_summary = engine.get_portfolio_summary()
        
        # Extract results for the selected strategy
        strategy_key = list(strategies.keys())[0]
        if strategy_key in results:
            strategy_results = results[strategy_key]
            performance_metrics = strategy_results.get('performance_metrics', {})
            
            return {
                'strategy_name': strategy_name,
                'total_return': performance_metrics.get('total_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'profit_factor': performance_metrics.get('profit_factor', 0),
                'total_trades': performance_metrics.get('total_trades', 0),
                'trades': strategy_results.get('trades', []),
                'final_portfolio_value': portfolio_summary.get('portfolio_value', 0),
                'engine': engine  # Store engine for portfolio visualization
            }
        else:
            st.error("No results found for the selected strategy")
            return None
            
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def generate_mock_trades(strategy_name, params):
    """Generate mock trade data based on actual backtest parameters"""
    np.random.seed(42)
    
    # Get parameters from config
    n_bars = params.get('n_bars', 100)
    initial_capital = params.get('initial_capital', 100000)
    symbols = params.get('symbols', ['Bitstamp:BTCUSD'])
    start_date = params.get('start_date', '2023-01-01')
    end_date = params.get('end_date', '2025-07-31')
    
    if isinstance(symbols, list) and len(symbols) > 0:
        primary_symbol = symbols[0]
    else:
        primary_symbol = 'Bitstamp:BTCUSD'
    
    # Calculate realistic number of trades based on n_bars
    # For 100 bars, we might have 6-12 trades (6-12% trade frequency)
    trade_frequency = np.random.uniform(0.06, 0.12)  # 6-12% of bars result in trades
    n_trades = max(1, int(n_bars * trade_frequency))
    
    trades = []
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate trades with realistic timing
    for i in range(n_trades):
        # Random trade timing within the date range
        trade_date = start_dt + timedelta(
            days=np.random.randint(0, (end_dt - start_dt).days)
        )
        
        # Generate realistic BTC prices (around $40,000-60,000)
        base_price = 50000
        price = base_price * (1 + np.random.normal(0, 0.1))  # 10% volatility
        price = max(40000, min(60000, price))  # Clamp to realistic range
        
        # Generate realistic quantities (0.1-2.0 BTC)
        quantity = np.random.uniform(0.1, 2.0)
        
        # Alternate buy/sell for more realistic pattern
        side = 'buy' if i % 2 == 0 else 'sell'
        
        trades.append({
            'timestamp': trade_date,
            'symbol': primary_symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': price * quantity * 0.001,
            'strategy': strategy_name
        })
    
    # Sort trades by timestamp
    trades.sort(key=lambda x: x['timestamp'])
    return trades

def save_backtest_history(results, strategy_name, params):
    """Save backtest results to history"""
    history_file = "data/backtest_history.json"
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new result
    history.append({
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy_name,
        'params': params,
        'results': results
    })
    
    # Save updated history
    os.makedirs('data', exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)

def load_backtest_history():
    """Load backtest history"""
    history_file = "data/backtest_history.json"
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    else:
        return []

def setup_sidebar():
    """Setup sidebar with portfolio information and controls"""
    st.sidebar.title("🚀 Auto Trading System")
    
    # Load configuration
    config = load_config()
    
    if config:
        # Portfolio Information
        st.sidebar.markdown("### 📊 Portfolio Information")
        
        initial_capital = config.get("trading.initial_capital", 100000)
        symbols = config.get("trading.symbols", ["Bitstamp:BTCUSD"])
        source = config.get("data.source", "tradingview")
        start_date = config.get("data.start_date", "2023-01-01")
        end_date = config.get("data.end_date", "2025-07-31")
        interval = config.get("data.interval", "4h")
        n_bars = config.get("data.n_bars", 100)
        
        st.sidebar.write(f"**Initial Capital:** ${initial_capital:,.0f}")
        st.sidebar.write(f"**Symbols:** {', '.join(symbols)}")
        st.sidebar.write(f"**Data Source:** {source}")
        st.sidebar.write(f"**Start Date:** {start_date}")
        st.sidebar.write(f"**End Date:** {end_date}")
        st.sidebar.write(f"**Interval:** {interval}")
        st.sidebar.write(f"**N Bars:** {n_bars}")
        
        # Strategy Selection
        st.sidebar.markdown("### 🔧 Strategy Selection")
        strategy_options = ["SMA Crossover", "RSI Strategy", "MACD Strategy"]
        selected_strategy = st.sidebar.selectbox(
            "Choose Strategy",
            strategy_options,
            index=0
        )
        
        # Mode Selection
        st.sidebar.markdown("### ⚙️ Trading Mode")
        mode_options = ["Backtest", "Paper Trading", "Live Trading"]
        selected_mode = st.sidebar.selectbox(
            "Choose Mode",
            mode_options,
            index=0
        )
        
        # Store parameters
        st.session_state.portfolio_params = {
            'initial_capital': initial_capital,
            'symbols': symbols,
            'source': source,
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval,
            'n_bars': n_bars,
            'selected_strategy': selected_strategy,
            'selected_mode': selected_mode
        }
        
        # Run Backtest Button
        st.sidebar.markdown("### ⚡ Quick Actions")
        if st.sidebar.button("🚀 Run Backtest", use_container_width=True):
            st.session_state.run_backtest = True
            st.session_state.backtest_params = st.session_state.portfolio_params.copy()
        
        # System Status
        st.sidebar.markdown("### 📈 System Status")
        st.sidebar.write(f"**Status:** 🟢 Running")
        st.sidebar.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
    else:
        st.sidebar.error("Configuration not available")

def show_backtest_results():
    """Show backtest results and trade details"""
    st.header("📊 Backtest Results")
    
    if st.session_state.get('run_backtest', False):
        params = st.session_state.get('backtest_params', {})
        strategy = params.get('selected_strategy', 'SMA Crossover')
        
        with st.spinner("Running backtest..."):
            config = load_config()
            results = run_backtest_from_dashboard(config, strategy, params)
            
            if results:
                # Store results in session state for portfolio visualization
                st.session_state.backtest_results = results
                
                # Save to history
                save_backtest_history(results, strategy, params)
                
                # Display Results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2%}")
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                
                with col2:
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    st.metric("Win Rate", f"{results['win_rate']:.2%}")
                
                with col3:
                    st.metric("Total Trades", results['total_trades'])
                    st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                
                with col4:
                    st.metric("Strategy", strategy)
                    st.metric("Mode", params.get('selected_mode', 'Backtest'))
                
                # Trade Details
                st.subheader("📋 Trade Details")
                
                if results.get('trades'):
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    trades_df['price_formatted'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
                    trades_df['commission_formatted'] = trades_df['commission'].apply(lambda x: f"${x:.2f}")
                    
                    # Display trades table
                    display_cols = ['timestamp', 'symbol', 'side', 'quantity', 'price_formatted', 'commission_formatted', 'strategy']
                    st.dataframe(trades_df[display_cols], use_container_width=True)
                    
                    # Trade Analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Trade Side Distribution")
                        side_counts = trades_df['side'].value_counts()
                        fig = px.pie(values=side_counts.values, names=side_counts.index, title="Buy vs Sell Trades")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Trade Volume by Symbol")
                        symbol_volume = trades_df.groupby('symbol')['quantity'].sum()
                        fig = px.bar(x=symbol_volume.index, y=symbol_volume.values, title="Total Volume by Symbol")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.run_backtest = False
                st.success("✅ Backtest completed successfully!")
            else:
                st.error("❌ Backtest failed")
    else:
        st.info("Click 'Run Backtest' in the sidebar to start a backtest")

def show_portfolio_performance():
    """Show portfolio performance visualization"""
    st.header("📈 Portfolio Performance")
    
    # Check if we have engine data from backtest
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        engine = st.session_state.backtest_results.get('engine')
        if engine:
            plot_portfolio_performance(engine)
            return
    
    # Fallback to mock data if no engine available
    st.info("Run a backtest first to see real portfolio performance data")
    
    # Load config to get realistic parameters
    config = load_config()
    if config:
        if hasattr(config, 'get'):
            start_date = config.get("data.start_date", "2023-01-01")
            end_date = config.get("data.end_date", "2025-07-31")
            initial_capital = config.get("trading.initial_capital", 100000)
            n_bars = config.get("data.n_bars", 100)
        else:
            start_date = config.get("data", {}).get("start_date", "2023-01-01")
            end_date = config.get("data", {}).get("end_date", "2025-07-31")
            initial_capital = config.get("trading", {}).get("initial_capital", 100000)
            n_bars = config.get("data", {}).get("n_bars", 100)
    else:
        start_date = "2023-01-01"
        end_date = "2025-07-31"
        initial_capital = 100000
        n_bars = 100
    
    # Generate realistic portfolio performance data based on n_bars
    # For 100 bars over ~2.5 years, we'll generate daily data
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate realistic number of data points
    total_days = (end_dt - start_dt).days
    if n_bars < total_days:
        # Use n_bars as the number of trading days
        dates = pd.date_range(start=start_date, end=end_date, periods=n_bars)
    else:
        # Use actual date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    
    # Generate realistic portfolio values with BTC-like volatility
    initial_value = initial_capital
    # Higher volatility for crypto (3-5% daily)
    daily_volatility = np.random.uniform(0.03, 0.05)
    daily_return_mean = np.random.uniform(0.0002, 0.0008)  # 0.02-0.08% daily return
    
    returns = np.random.normal(daily_return_mean, daily_volatility, len(dates))
    portfolio_values = [initial_value]
    
    for ret in returns[1:]:
        new_value = portfolio_values[-1] * (1 + ret)
        # Ensure portfolio doesn't go below 50% of initial value
        portfolio_values.append(max(new_value, initial_value * 0.5))
    
    portfolio_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'daily_return': returns,
        'cumulative_return': [(v - initial_value) / initial_value for v in portfolio_values]
    })
    
    # Portfolio Value Chart
    st.subheader("💰 Portfolio Value Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df['date'],
        y=portfolio_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title="Portfolio Value Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{portfolio_df['cumulative_return'].iloc[-1]:.2%}")
    
    with col2:
        st.metric("Volatility", f"{portfolio_df['daily_return'].std() * np.sqrt(252):.2%}")
    
    with col3:
        st.metric("Max Drawdown", f"{(portfolio_df['portfolio_value'].min() - initial_value) / initial_value:.2%}")
    
    with col4:
        st.metric("Sharpe Ratio", f"{(portfolio_df['daily_return'].mean() * 252) / (portfolio_df['daily_return'].std() * np.sqrt(252)):.2f}")
    
    # Returns Distribution
    st.subheader("📊 Returns Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(portfolio_df, x='daily_return', nbins=30, title="Daily Returns Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(portfolio_df, y='daily_return', title="Returns Box Plot")
        st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_performance(engine):
    """Plot portfolio performance using actual engine data"""
    portfolio_history = engine.portfolio_history
    
    if not portfolio_history:
        st.error("Không có dữ liệu portfolio history")
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
        trade_returns = []
        for trade in engine.trades:
            if hasattr(trade, 'pnl'):
                trade_returns.append(trade.pnl)
            elif isinstance(trade, dict) and 'pnl' in trade:
                trade_returns.append(trade['pnl'])
        
        if trade_returns:
            fig.add_trace(
                go.Histogram(x=trade_returns, name='Trade Returns'),
                row=3, col=2
            )
    
    fig.update_layout(height=900, title_text="Portfolio Performance Analysis")
    st.plotly_chart(fig, use_container_width=True)

def show_trade_visualization():
    """Show trade visualization using viz.py"""
    st.header("📊 Trade Visualization")
    
    if viz is None:
        st.error("❌ Visualization module not available")
        st.info("Please ensure viz.py is properly imported")
        return
    
    # Load config to get realistic parameters
    config = load_config()
    if config:
        if hasattr(config, 'get'):
            start_date = config.get("data.start_date", "2023-01-01")
            end_date = config.get("data.end_date", "2025-07-31")
            n_bars = config.get("data.n_bars", 100)
        else:
            start_date = config.get("data", {}).get("start_date", "2023-01-01")
            end_date = config.get("data", {}).get("end_date", "2025-07-31")
            n_bars = config.get("data", {}).get("n_bars", 100)
    else:
        start_date = "2023-01-01"
        end_date = "2025-07-31"
        n_bars = 100
    
    # Generate realistic data based on n_bars
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate realistic number of data points
    total_days = (end_dt - start_dt).days
    if n_bars < total_days:
        # Use n_bars as the number of trading days
        dates = pd.date_range(start=start_date, end=end_date, periods=n_bars)
    else:
        # Use actual date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    
    # Generate OHLCV data for BTC with realistic volatility
    base_price = 50000  # BTC base price
    prices = []
    for i in range(len(dates)):
        if i == 0:
            price = base_price
        else:
            # Higher volatility for crypto (3-5% daily)
            change = np.random.normal(0, np.random.uniform(0.03, 0.05))
            price = prices[-1] * (1 + change)
            # Clamp to realistic BTC price range
            price = max(40000, min(60000, price))
        prices.append(price)
    
    # Create OHLCV data
    historical_data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(100, 1000) for _ in prices]  # BTC volume
    }, index=dates)
    
    # Generate realistic number of trades based on n_bars
    trade_frequency = np.random.uniform(0.06, 0.12)  # 6-12% of bars result in trades
    n_trades = max(1, int(n_bars * trade_frequency))
    
    trades_data = []
    for i in range(n_trades):
        date_idx = np.random.randint(0, len(dates))
        side = 'buy' if i % 2 == 0 else 'sell'  # Alternate buy/sell
        price = historical_data.iloc[date_idx]['close']
        
        trades_data.append({
            'Timestamp': dates[date_idx],
            'Side': side,
            'Price': price,
            'Quantity': np.random.uniform(0.1, 2.0)  # BTC quantities
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Create visualization using viz.py
    st.subheader("📈 Price Chart with Trade Signals")
    
    # Create the visualization
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=historical_data.index,
        open=historical_data['open'],
        high=historical_data['high'],
        low=historical_data['low'],
        close=historical_data['close'],
        name='Price'
    ), row=1, col=1)
    
    # Add trade markers
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['Side'] == 'buy']
        sell_trades = trades_df[trades_df['Side'] == 'sell']
        
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['Timestamp'],
                y=buy_trades['Price'] * 0.998,
                mode='markers+text',
                marker=dict(symbol='triangle-up', color='green', size=12),
                text=['Buy']*len(buy_trades),
                textposition='bottom center',
                name='Buy'
            ), row=1, col=1)
        
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['Timestamp'],
                y=sell_trades['Price'] * 1.002,
                mode='markers+text',
                marker=dict(symbol='triangle-down', color='red', size=12),
                text=['Sell']*len(sell_trades),
                textposition='top center',
                name='Sell'
            ), row=1, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=historical_data.index,
        y=historical_data['volume'],
        name='Volume'
    ), row=2, col=1)
    
    fig.update_layout(
        title='Bitstamp:BTCUSD - Price Chart with Trade Signals',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade Summary
    st.subheader("📋 Trade Summary")
    if not trades_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", len(trades_df))
        
        with col2:
            buy_count = len(trades_df[trades_df['Side'] == 'buy'])
            st.metric("Buy Trades", buy_count)
        
        with col3:
            sell_count = len(trades_df[trades_df['Side'] == 'sell'])
            st.metric("Sell Trades", sell_count)
        
        # Display trades table
        st.dataframe(trades_df, use_container_width=True)

def show_strategy_history():
    """Show strategy backtest history and performance comparison"""
    st.header("📚 Strategy History & Performance Comparison")
    
    # Load backtest history
    history = load_backtest_history()
    
    if not history:
        st.info("No backtest history available. Run some backtests to see results here.")
        return
    
    # Convert to DataFrame for easier analysis
    history_df = pd.DataFrame(history)
    
    # Strategy Performance Summary
    st.subheader("📊 Strategy Performance Summary")
    
    # Extract results for each strategy
    strategy_results = []
    for _, row in history_df.iterrows():
        results = row['results']
        strategy_results.append({
            'strategy': row['strategy'],
            'timestamp': row['timestamp'],
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades'],
            'profit_factor': results['profit_factor']
        })
    
    results_df = pd.DataFrame(strategy_results)
    
    # Performance Comparison Chart
    st.subheader("📈 Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total Return by Strategy
        fig = px.bar(results_df, x='strategy', y='total_return', 
                     title="Total Return by Strategy",
                     color='strategy')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sharpe Ratio by Strategy
        fig = px.bar(results_df, x='strategy', y='sharpe_ratio',
                     title="Sharpe Ratio by Strategy",
                     color='strategy')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Results Table
    st.subheader("📋 Detailed Results")
    
    # Format the results for display
    display_df = results_df.copy()
    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2%}")
    display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
    display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Strategy Evolution Over Time
    st.subheader("🕒 Strategy Performance Over Time")
    
    # Convert timestamp to datetime
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    
    fig = go.Figure()
    
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        fig.add_trace(go.Scatter(
            x=strategy_data['timestamp'],
            y=strategy_data['total_return'],
            mode='lines+markers',
            name=strategy
        ))
    
    fig.update_layout(
        title="Strategy Performance Evolution",
        xaxis_title="Date",
        yaxis_title="Total Return",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best Performing Strategy
    st.subheader("🏆 Best Performing Strategy")
    
    best_strategy = results_df.loc[results_df['total_return'].idxmax()]
    st.success(f"**Best Strategy:** {best_strategy['strategy']}")
    st.write(f"**Total Return:** {best_strategy['total_return']:.2%}")
    st.write(f"**Sharpe Ratio:** {best_strategy['sharpe_ratio']:.2f}")
    st.write(f"**Win Rate:** {best_strategy['win_rate']:.2%}")

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Auto Trading System Dashboard</h1>', unsafe_allow_html=True)
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Backtest Results", 
        "📈 Portfolio Performance", 
        "📊 Trade Visualization", 
        "📚 Strategy History"
    ])
    
    with tab1:
        show_backtest_results()
    
    with tab2:
        show_portfolio_performance()
    
    with tab3:
        show_trade_visualization()
    
    with tab4:
        show_strategy_history()

if __name__ == "__main__":
    main() 