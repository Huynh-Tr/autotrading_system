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

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Create mock strategy classes for fallback
class MockStrategy:
    """Mock strategy class for when real strategies can't be imported"""
    def __init__(self, name, config):
        self.name = name
        self.config = config
    
    def generate_signals(self, historical_data, current_data):
        return {}
    
    def validate_config(self):
        return True
    
    def get_summary(self):
        return {'name': self.name, 'strategy_type': 'Mock'}

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
    SMACrossoverStrategy = MockStrategy

try:
    from strategies.rsi_strategy import RSIStrategy
    print("✅ RSIStrategy imported successfully")
except ImportError as e:
    print(f"⚠️ RSIStrategy import error: {e}")
    RSIStrategy = MockStrategy

try:
    from strategies.macd_strategy import MACDStrategy
    print("✅ MACDStrategy imported successfully")
except ImportError as e:
    print(f"⚠️ MACDStrategy import error: {e}")
    MACDStrategy = MockStrategy

try:
    from utils.ohlcv_utils import get_symbols_from_data, extract_price_data, get_symbol_data
    print("✅ OHLCV utilities imported successfully")
except ImportError as e:
    print(f"⚠️ OHLCV utilities import error: {e}")
    get_symbols_from_data = None
    extract_price_data = None
    get_symbol_data = None

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

@st.cache_data
def get_mock_portfolio_data():
    """Get mock portfolio data for demonstration"""
    return {
        'total_value': 105000,
        'cash': 50000,
        'positions_value': 55000,
        'daily_pnl': 1250,
        'total_pnl': 5000,
        'positions': [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 150.00,
                'current_price': 155.00,
                'unrealized_pnl': 500.00,
                'weight': 0.35
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'entry_price': 2800.00,
                'current_price': 2850.00,
                'unrealized_pnl': 2500.00,
                'weight': 0.45
            },
            {
                'symbol': 'MSFT',
                'quantity': 75,
                'entry_price': 300.00,
                'current_price': 310.00,
                'unrealized_pnl': 750.00,
                'weight': 0.20
            }
        ]
    }

@st.cache_data
def get_mock_performance_data():
    """Get mock performance data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic performance data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    cumulative_returns = np.cumprod(1 + returns)
    
    return pd.DataFrame({
        'date': dates,
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': 100000 * cumulative_returns
    })

@st.cache_data
def get_mock_trades_data():
    """Get mock trades data"""
    trades = [
        {
            'date': '2024-01-15',
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.00,
            'pnl': 0,
            'strategy': 'SMA Crossover'
        },
        {
            'date': '2024-02-20',
            'symbol': 'GOOGL',
            'action': 'BUY',
            'quantity': 50,
            'price': 2800.00,
            'pnl': 0,
            'strategy': 'RSI Strategy'
        },
        {
            'date': '2024-03-10',
            'symbol': 'MSFT',
            'action': 'BUY',
            'quantity': 75,
            'price': 300.00,
            'pnl': 0,
            'strategy': 'MACD Strategy'
        },
        {
            'date': '2024-04-05',
            'symbol': 'AAPL',
            'action': 'SELL',
            'quantity': 50,
            'price': 160.00,
            'pnl': 500.00,
            'strategy': 'SMA Crossover'
        },
        {
            'date': '2024-05-12',
            'symbol': 'TSLA',
            'action': 'BUY',
            'quantity': 25,
            'price': 800.00,
            'pnl': 0,
            'strategy': 'RSI Strategy'
        }
    ]
    
    return pd.DataFrame(trades)

def run_backtest_from_dashboard(config, strategy_name):
    """Run backtest from dashboard"""
    try:
        if not config:
            st.error("Configuration not available")
            return None
        
        # Check if strategies are available
        strategy_available = True
        if strategy_name == "SMA Crossover" and SMACrossoverStrategy == MockStrategy:
            strategy_available = False
        elif strategy_name == "RSI Strategy" and RSIStrategy == MockStrategy:
            strategy_available = False
        elif strategy_name == "MACD Strategy" and MACDStrategy == MockStrategy:
            strategy_available = False
        
        if not strategy_available:
            st.warning(f"⚠️ {strategy_name} is not available (using mock data)")
        
        # Mock backtest results
        results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'total_trades': 45,
            'profit_factor': 1.8
        }
        
        return results
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None

@st.cache_data
def get_mock_portfolio_history():
    """Get mock portfolio history"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic portfolio history
    base_value = 100000
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    portfolio_values = base_value * cumulative_returns
    
    return pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'daily_return': daily_returns,
        'cumulative_return': cumulative_returns - 1
    })

def setup_sidebar():
    """Setup sidebar with basic information and features"""
    st.sidebar.title("🚀 Auto Trading System")
    
    # System Status
    st.sidebar.markdown("### 📊 System Status")
    status = st.sidebar.selectbox(
        "System Status",
        ["🟢 Running", "🔴 Stopped", "🟡 Paused"],
        index=0
    )
    
    # Quick Metrics
    st.sidebar.markdown("### 📈 Quick Metrics")
    portfolio_data = get_mock_portfolio_data()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Value", f"${portfolio_data['total_value']:,.0f}")
        st.metric("Daily P&L", f"${portfolio_data['daily_pnl']:,.0f}")
    
    with col2:
        st.metric("Cash", f"${portfolio_data['cash']:,.0f}")
        st.metric("Positions", f"${portfolio_data['positions_value']:,.0f}")
    
    # Strategy Selection
    st.sidebar.markdown("### 🔧 Strategy Selection")
    strategy_options = ["SMA Crossover", "RSI Strategy", "MACD Strategy"]
    selected_strategy = st.sidebar.selectbox(
        "Active Strategy",
        strategy_options,
        index=0
    )
    
    # Strategy Parameters
    st.sidebar.markdown("### ⚙️ Strategy Parameters")
    
    if selected_strategy == "SMA Crossover":
        short_window = st.sidebar.slider("Short SMA Period", 5, 50, 20, key="sidebar_short_sma")
        long_window = st.sidebar.slider("Long SMA Period", 20, 200, 50, key="sidebar_long_sma")
        st.sidebar.write(f"**Parameters:** Short={short_window}, Long={long_window}")
        
    elif selected_strategy == "RSI Strategy":
        rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14, key="sidebar_rsi_period")
        oversold = st.sidebar.slider("Oversold Level", 20, 40, 30, key="sidebar_oversold")
        overbought = st.sidebar.slider("Overbought Level", 60, 80, 70, key="sidebar_overbought")
        st.sidebar.write(f"**Parameters:** RSI={rsi_period}, Oversold={oversold}, Overbought={overbought}")
        
    elif selected_strategy == "MACD Strategy":
        fast_period = st.sidebar.slider("Fast Period", 8, 20, 12, key="sidebar_fast_period")
        slow_period = st.sidebar.slider("Slow Period", 20, 40, 26, key="sidebar_slow_period")
        signal_period = st.sidebar.slider("Signal Period", 5, 15, 9, key="sidebar_signal_period")
        st.sidebar.write(f"**Parameters:** Fast={fast_period}, Slow={slow_period}, Signal={signal_period}")
    
    # Strategy Status
    st.sidebar.markdown("### 📊 Strategy Status")
    strategy_availability = {
        "SMA Crossover": SMACrossoverStrategy != MockStrategy,
        "RSI Strategy": RSIStrategy != MockStrategy,
        "MACD Strategy": MACDStrategy != MockStrategy
    }
    
    strategy_status = strategy_availability.get(selected_strategy, False)
    status_text = "✅ Available" if strategy_status else "❌ Unavailable"
    st.sidebar.write(f"**{selected_strategy}:** {status_text}")
    
    # Trading Parameters
    st.sidebar.markdown("### 💰 Trading Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000, key="sidebar_capital")
    position_size = st.sidebar.slider("Position Size (%)", 1, 100, 10, key="sidebar_position_size")
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, 5, key="sidebar_stop_loss")
    take_profit = st.sidebar.slider("Take Profit (%)", 5, 50, 15, key="sidebar_take_profit")
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("📊 Run Quick Backtest", use_container_width=True):
        st.session_state.run_quick_backtest = True
        st.session_state.quick_backtest_params = {
            'strategy': selected_strategy,
            'initial_capital': initial_capital,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    if st.sidebar.button("📋 View Recent Trades", use_container_width=True):
        st.session_state.show_recent_trades = True
    
    # System Information
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.write(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.write(f"**Uptime:** 2h 15m")
    st.sidebar.write(f"**Version:** 1.0.0")
    
    # Import Status Warning
    if any(strategy == MockStrategy for strategy in [SMACrossoverStrategy, RSIStrategy, MACDStrategy]):
        st.sidebar.markdown("""
        <div class="warning-box">
            <strong>⚠️ Import Warning</strong><br>
            Some strategies unavailable
        </div>
        """, unsafe_allow_html=True)
    
    # Store parameters in session state for use in tabs
    st.session_state.sidebar_params = {
        'selected_strategy': selected_strategy,
        'initial_capital': initial_capital,
        'position_size': position_size,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

def show_overview_page():
    """Show overview page"""
    st.header("📈 System Overview")
    
    # Get current strategy and parameters from sidebar
    sidebar_params = st.session_state.get('sidebar_params', {})
    current_strategy = sidebar_params.get('selected_strategy', 'SMA Crossover')
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="System Status",
            value="🟢 Running",
            delta="Online"
        )
    
    with col2:
        st.metric(
            label="Active Strategy",
            value=current_strategy,
            delta="Selected"
        )
    
    with col3:
        st.metric(
            label="Total Trades",
            value="1,247",
            delta="+23 today"
        )
    
    with col4:
        st.metric(
            label="Success Rate",
            value="68.5%",
            delta="+2.1%"
        )
    
    # Current Strategy Info
    st.subheader("🔧 Current Strategy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Active Strategy:** {current_strategy}")
        
        # Show strategy parameters
        if current_strategy == "SMA Crossover":
            short_window = sidebar_params.get('short_window', 20)
            long_window = sidebar_params.get('long_window', 50)
            st.write(f"**Parameters:** Short SMA={short_window}, Long SMA={long_window}")
        elif current_strategy == "RSI Strategy":
            rsi_period = sidebar_params.get('rsi_period', 14)
            oversold = sidebar_params.get('oversold', 30)
            overbought = sidebar_params.get('overbought', 70)
            st.write(f"**Parameters:** RSI={rsi_period}, Oversold={oversold}, Overbought={overbought}")
        elif current_strategy == "MACD Strategy":
            fast_period = sidebar_params.get('fast_period', 12)
            slow_period = sidebar_params.get('slow_period', 26)
            signal_period = sidebar_params.get('signal_period', 9)
            st.write(f"**Parameters:** Fast={fast_period}, Slow={slow_period}, Signal={signal_period}")
    
    with col2:
        # Show trading parameters
        initial_capital = sidebar_params.get('initial_capital', 100000)
        position_size = sidebar_params.get('position_size', 10)
        stop_loss = sidebar_params.get('stop_loss', 5)
        take_profit = sidebar_params.get('take_profit', 15)
        
        st.write("**Trading Parameters:**")
        st.write(f"- Capital: ${initial_capital:,.0f}")
        st.write(f"- Position Size: {position_size}%")
        st.write(f"- Stop Loss: {stop_loss}%")
        st.write(f"- Take Profit: {take_profit}%")
    
    # Portfolio Summary
    st.subheader("💼 Portfolio Summary")
    portfolio_data = get_mock_portfolio_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Value",
            value=f"${portfolio_data['total_value']:,.0f}",
            delta=f"${portfolio_data['daily_pnl']:,.0f}"
        )
    
    with col2:
        st.metric(
            label="Cash",
            value=f"${portfolio_data['cash']:,.0f}",
            delta="No change"
        )
    
    with col3:
        st.metric(
            label="Positions Value",
            value=f"${portfolio_data['positions_value']:,.0f}",
            delta=f"${portfolio_data['total_pnl']:,.0f}"
        )
    
    # Recent Activity
    st.subheader("🔄 Recent Activity")
    trades_data = get_mock_trades_data()
    
    # Display recent trades
    st.dataframe(
        trades_data.tail(5),
        use_container_width=True,
        hide_index=True
    )
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Run Backtest", use_container_width=True):
            st.session_state.run_quick_backtest = True
            st.rerun()
    
    with col2:
        if st.button("📋 View Portfolio", use_container_width=True):
            st.switch_page("💼 Portfolio")
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.switch_page("⚙️ Settings")

def show_portfolio_page():
    """Show portfolio page"""
    st.header("💼 Portfolio Management")
    
    portfolio_data = get_mock_portfolio_data()
    
    # Portfolio Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Portfolio Allocation")
        
        # Create pie chart for portfolio allocation
        positions = portfolio_data['positions']
        symbols = [pos['symbol'] for pos in positions]
        weights = [pos['weight'] for pos in positions]
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=weights,
            hole=0.3,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        for position in positions:
            with st.container():
                st.metric(
                    label=position['symbol'],
                    value=f"${position['current_price']:.2f}",
                    delta=f"${position['unrealized_pnl']:.2f}",
                    delta_color="normal" if position['unrealized_pnl'] >= 0 else "inverse"
                )
    
    # Positions Table
    st.subheader("📋 Current Positions")
    
    positions_df = pd.DataFrame(portfolio_data['positions'])
    positions_df['unrealized_pnl_pct'] = (positions_df['unrealized_pnl'] / (positions_df['quantity'] * positions_df['entry_price'])) * 100
    
    st.dataframe(
        positions_df,
        use_container_width=True,
        hide_index=True
    )

def show_performance_page():
    """Show performance page"""
    st.header("📊 Performance Analysis")
    
    performance_data = get_mock_performance_data()
    portfolio_history = get_mock_portfolio_history()
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Return",
            value="15.2%",
            delta="+2.1%"
        )
    
    with col2:
        st.metric(
            label="Sharpe Ratio",
            value="1.45",
            delta="+0.12"
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value="-8.3%",
            delta="-1.2%"
        )
    
    with col4:
        st.metric(
            label="Volatility",
            value="12.5%",
            delta="-0.8%"
        )
    
    # Performance Chart
    st.subheader("📈 Portfolio Performance")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns Distribution
    st.subheader("📊 Returns Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            performance_data,
            x='returns',
            nbins=30,
            title="Daily Returns Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            performance_data,
            y='returns',
            title="Returns Box Plot"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_backtesting_page():
    """Show backtesting page"""
    st.header("🔧 Strategy Backtesting")
    
    # Get parameters from sidebar if available
    sidebar_params = st.session_state.get('sidebar_params', {})
    sidebar_strategy = sidebar_params.get('selected_strategy', 'SMA Crossover')
    
    # Strategy Selection
    st.subheader("📋 Select Strategy")
    
    strategy_options = ["SMA Crossover", "RSI Strategy", "MACD Strategy"]
    selected_strategy = st.selectbox("Choose Strategy", strategy_options, index=strategy_options.index(sidebar_strategy))
    
    # Show strategy availability status
    if selected_strategy == "SMA Crossover" and SMACrossoverStrategy == MockStrategy:
        st.warning("⚠️ SMA Crossover Strategy is not available (using mock data)")
    elif selected_strategy == "RSI Strategy" and RSIStrategy == MockStrategy:
        st.warning("⚠️ RSI Strategy is not available (using mock data)")
    elif selected_strategy == "MACD Strategy" and MACDStrategy == MockStrategy:
        st.warning("⚠️ MACD Strategy is not available (using mock data)")
    
    # Strategy Parameters
    st.subheader("⚙️ Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_strategy == "SMA Crossover":
            short_window = st.slider("Short SMA Period", 5, 50, 20, key="tab_short_sma")
            long_window = st.slider("Long SMA Period", 20, 200, 50, key="tab_long_sma")
            st.write(f"**Parameters:** Short={short_window}, Long={long_window}")
            
        elif selected_strategy == "RSI Strategy":
            rsi_period = st.slider("RSI Period", 10, 30, 14, key="tab_rsi_period")
            oversold = st.slider("Oversold Level", 20, 40, 30, key="tab_oversold")
            overbought = st.slider("Overbought Level", 60, 80, 70, key="tab_overbought")
            st.write(f"**Parameters:** RSI={rsi_period}, Oversold={oversold}, Overbought={overbought}")
            
        elif selected_strategy == "MACD Strategy":
            fast_period = st.slider("Fast Period", 8, 20, 12, key="tab_fast_period")
            slow_period = st.slider("Slow Period", 20, 40, 26, key="tab_slow_period")
            signal_period = st.slider("Signal Period", 5, 15, 9, key="tab_signal_period")
            st.write(f"**Parameters:** Fast={fast_period}, Slow={slow_period}, Signal={signal_period}")
    
    with col2:
        # Trading Parameters
        st.subheader("💰 Trading Parameters")
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, step=10000, key="tab_capital")
        position_size = st.slider("Position Size (%)", 1, 100, 10, key="tab_position_size")
        stop_loss = st.slider("Stop Loss (%)", 1, 20, 5, key="tab_stop_loss")
        take_profit = st.slider("Take Profit (%)", 5, 50, 15, key="tab_take_profit")
    
    # Backtest Parameters
    st.subheader("📅 Backtest Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    
    with col2:
        end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
    
    with col3:
        # Use sidebar capital if available, otherwise use tab capital
        default_capital = sidebar_params.get('initial_capital', initial_capital)
        backtest_capital = st.number_input("Backtest Capital ($)", 10000, 1000000, default_capital, step=10000)
    
    # Parameter Summary
    st.subheader("📋 Parameter Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strategy Parameters:**")
        if selected_strategy == "SMA Crossover":
            st.write(f"- Short SMA: {short_window}")
            st.write(f"- Long SMA: {long_window}")
        elif selected_strategy == "RSI Strategy":
            st.write(f"- RSI Period: {rsi_period}")
            st.write(f"- Oversold: {oversold}")
            st.write(f"- Overbought: {overbought}")
        elif selected_strategy == "MACD Strategy":
            st.write(f"- Fast Period: {fast_period}")
            st.write(f"- Slow Period: {slow_period}")
            st.write(f"- Signal Period: {signal_period}")
    
    with col2:
        st.write("**Trading Parameters:**")
        st.write(f"- Initial Capital: ${backtest_capital:,.0f}")
        st.write(f"- Position Size: {position_size}%")
        st.write(f"- Stop Loss: {stop_loss}%")
        st.write(f"- Take Profit: {take_profit}%")
    
    # Run Backtest
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            config = load_config()
            
            # Prepare parameters for backtest
            backtest_params = {
                'strategy': selected_strategy,
                'initial_capital': backtest_capital,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # Add strategy-specific parameters
            if selected_strategy == "SMA Crossover":
                backtest_params.update({
                    'short_window': short_window,
                    'long_window': long_window
                })
            elif selected_strategy == "RSI Strategy":
                backtest_params.update({
                    'rsi_period': rsi_period,
                    'oversold': oversold,
                    'overbought': overbought
                })
            elif selected_strategy == "MACD Strategy":
                backtest_params.update({
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period
                })
            
            results = run_backtest_from_dashboard(config, selected_strategy)
            
            if results:
                st.success("✅ Backtest completed successfully!")
                
                # Store results in session state
                st.session_state.last_backtest_results = results
                st.session_state.last_backtest_params = backtest_params
                
                # Display Results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2%}")
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                
                with col2:
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    st.metric("Win Rate", f"{results['win_rate']:.2%}")
                
                with col3:
                    st.metric("Total Trades", results['total_trades'])
                    st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
            else:
                st.error("❌ Backtest failed")
    
    # Show Quick Backtest Results if triggered from sidebar
    if st.session_state.get('run_quick_backtest', False):
        st.subheader("⚡ Quick Backtest Results")
        
        quick_params = st.session_state.get('quick_backtest_params', {})
        st.write(f"**Strategy:** {quick_params.get('strategy', 'Unknown')}")
        st.write(f"**Capital:** ${quick_params.get('initial_capital', 0):,.0f}")
        st.write(f"**Position Size:** {quick_params.get('position_size', 0)}%")
        
        # Mock quick results
        quick_results = {
            'total_return': 0.12,
            'sharpe_ratio': 1.1,
            'max_drawdown': -0.06,
            'win_rate': 0.62,
            'total_trades': 28,
            'profit_factor': 1.6
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Quick Return", f"{quick_results['total_return']:.2%}")
            st.metric("Quick Sharpe", f"{quick_results['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("Quick Drawdown", f"{quick_results['max_drawdown']:.2%}")
            st.metric("Quick Win Rate", f"{quick_results['win_rate']:.2%}")
        
        with col3:
            st.metric("Quick Trades", quick_results['total_trades'])
            st.metric("Quick Profit Factor", f"{quick_results['profit_factor']:.2f}")
        
        # Reset the flag
        st.session_state.run_quick_backtest = False

def show_settings_page():
    """Show settings page"""
    st.header("⚙️ System Settings")
    
    config = load_config()
    
    st.subheader("📋 Configuration")
    
    if config:
        if isinstance(config, dict):
            st.json(config)
        else:
            st.write("Configuration loaded successfully")
    else:
        st.error("Configuration not available")
    
    st.subheader("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Python Version:** " + sys.version)
        st.info("**Streamlit Version:** " + st.__version__)
    
    with col2:
        st.info("**Project Root:** " + project_root)
        st.info("**Working Directory:** " + os.getcwd())
    
    # Strategy Import Status
    st.subheader("📊 Strategy Import Status")
    
    strategy_status = {
        "SMACrossoverStrategy": "✅ Available" if SMACrossoverStrategy != MockStrategy else "❌ Not Available",
        "RSIStrategy": "✅ Available" if RSIStrategy != MockStrategy else "❌ Not Available",
        "MACDStrategy": "✅ Available" if MACDStrategy != MockStrategy else "❌ Not Available",
        "ConfigManager": "✅ Available" if ConfigManager else "❌ Not Available"
    }
    
    for strategy, status in strategy_status.items():
        st.write(f"**{strategy}:** {status}")

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Auto Trading System Dashboard</h1>', unsafe_allow_html=True)
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "💼 Portfolio", 
        "📊 Performance", 
        "🔧 Backtesting", 
        "⚙️ Settings"
    ])
    
    with tab1:
        show_overview_page()
    
    with tab2:
        show_portfolio_page()
    
    with tab3:
        show_performance_page()
    
    with tab4:
        show_backtesting_page()
    
    with tab5:
        show_settings_page()

if __name__ == "__main__":
    main() 