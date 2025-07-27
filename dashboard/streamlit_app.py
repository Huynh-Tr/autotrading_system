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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.utils.config_manager import ConfigManager
    from src.core.trading_engine import TradingEngine
    from src.strategies.sma_crossover import SMACrossoverStrategy
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please run the setup script first: python setup.py")

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration with caching"""
    try:
        return ConfigManager("config/config.yaml")
    except:
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
                'pnl': 500,
                'pnl_pct': 3.33
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'entry_price': 2800.00,
                'current_price': 2850.00,
                'pnl': 2500,
                'pnl_pct': 1.79
            },
            {
                'symbol': 'MSFT',
                'quantity': 75,
                'entry_price': 300.00,
                'current_price': 310.00,
                'pnl': 750,
                'pnl_pct': 2.5
            }
        ]
    }

@st.cache_data
def get_mock_performance_data():
    """Get mock performance data"""
    return {
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'volatility': 0.15,
        'win_rate': 0.65,
        'total_trades': 25,
        'avg_trade_duration': 5.2
    }

@st.cache_data
def get_mock_trades_data():
    """Get mock trades data"""
    return [
        {
            'timestamp': '2024-01-15 10:30:00',
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'price': 150.00,
            'pnl': 0,
            'strategy': 'SMA Crossover'
        },
        {
            'timestamp': '2024-01-14 14:45:00',
            'symbol': 'GOOGL',
            'side': 'buy',
            'quantity': 50,
            'price': 2800.00,
            'pnl': 0,
            'strategy': 'SMA Crossover'
        },
        {
            'timestamp': '2024-01-13 11:20:00',
            'symbol': 'MSFT',
            'side': 'buy',
            'quantity': 75,
            'price': 300.00,
            'pnl': 0,
            'strategy': 'SMA Crossover'
        },
        {
            'timestamp': '2024-01-12 16:15:00',
            'symbol': 'TSLA',
            'side': 'sell',
            'quantity': 25,
            'price': 250.00,
            'pnl': 1250,
            'strategy': 'SMA Crossover'
        }
    ]

@st.cache_data
def get_mock_portfolio_history():
    """Get mock portfolio history for charts"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
    np.random.seed(42)
    
    # Generate realistic portfolio values
    initial_value = 100000
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    portfolio_values = [initial_value]
    
    for ret in returns[1:]:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(max(new_value, 50000))  # Minimum value
    
    return pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'cash': [50000] * len(dates),  # Mock cash values
        'positions_value': [pv - 50000 for pv in portfolio_values]
    })

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Auto Trading System Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # System status
        status = st.selectbox(
            "System Status",
            ["Running", "Stopped", "Paused"],
            index=0
        )
        
        st.markdown(f"**Status:** <span class='status-{status.lower()}'>● {status}</span>", unsafe_allow_html=True)
        
        # Trading mode
        mode = st.selectbox(
            "Trading Mode",
            ["Backtest", "Paper Trading", "Live Trading"],
            index=0
        )
        
        # Strategy selection
        strategy = st.selectbox(
            "Active Strategy",
            ["SMA Crossover", "RSI Strategy", "MACD Strategy"],
            index=0
        )
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📊 Run Backtest"):
                st.info("Backtest started...")
        
        st.divider()
        
        # System info
        st.header("ℹ️ System Info")
        st.write("**Last Update:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Uptime:** 2 hours 15 minutes")
        st.write("**Version:** 1.0.0")
    
    # Load data
    config = load_config()
    portfolio_data = get_mock_portfolio_data()
    performance_data = get_mock_performance_data()
    trades_data = get_mock_trades_data()
    portfolio_history = get_mock_portfolio_history()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance", "💼 Positions", "📋 Trades"])
    
    with tab1:
        # Portfolio Overview
        st.header("💰 Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=f"${portfolio_data['total_value']:,.0f}",
                delta=f"${portfolio_data['total_pnl']:,.0f} ({portfolio_data['total_pnl']/100000*100:.1f}%)"
            )
        
        with col2:
            st.metric(
                label="Available Cash",
                value=f"${portfolio_data['cash']:,.0f}",
                delta="Available for trading"
            )
        
        with col3:
            st.metric(
                label="Daily P&L",
                value=f"${portfolio_data['daily_pnl']:,.0f}",
                delta=f"{portfolio_data['daily_pnl']/portfolio_data['total_value']*100:.2f}%"
            )
        
        with col4:
            st.metric(
                label="Total Trades",
                value=performance_data['total_trades'],
                delta=f"Win Rate: {performance_data['win_rate']*100:.0f}%"
            )
        
        # Portfolio Performance Chart
        st.subheader("📈 Portfolio Performance")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_history['date'],
            y=portfolio_history['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=portfolio_history['date'],
            y=portfolio_history['cash'],
            mode='lines',
            name='Cash',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            
            metrics_df = pd.DataFrame([
                ["Total Return", f"{performance_data['total_return']*100:.2f}%"],
                ["Sharpe Ratio", f"{performance_data['sharpe_ratio']:.2f}"],
                ["Max Drawdown", f"{performance_data['max_drawdown']*100:.2f}%"],
                ["Volatility", f"{performance_data['volatility']*100:.2f}%"],
                ["Win Rate", f"{performance_data['win_rate']*100:.0f}%"],
                ["Avg Trade Duration", f"{performance_data['avg_trade_duration']:.1f} days"]
            ], columns=["Metric", "Value"])
            
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Metrics")
            
            # Risk gauge charts
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=("Sharpe Ratio", "Max Drawdown", "Win Rate", "Volatility")
            )
            
            # Sharpe Ratio gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data['sharpe_ratio'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe"},
                gauge={'axis': {'range': [None, 3]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 1], 'color': "lightgray"},
                                {'range': [1, 2], 'color': "yellow"},
                                {'range': [2, 3], 'color': "green"}]}
            ), row=1, col=1)
            
            # Max Drawdown gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=abs(performance_data['max_drawdown']*100),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Drawdown %"},
                gauge={'axis': {'range': [None, 20]},
                       'bar': {'color': "red"},
                       'steps': [{'range': [0, 5], 'color': "green"},
                                {'range': [5, 10], 'color': "yellow"},
                                {'range': [10, 20], 'color': "red"}]}
            ), row=1, col=2)
            
            # Win Rate gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data['win_rate']*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Win Rate %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "green"},
                       'steps': [{'range': [0, 50], 'color': "red"},
                                {'range': [50, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}]}
            ), row=2, col=1)
            
            # Volatility gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=performance_data['volatility']*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Volatility %"},
                gauge={'axis': {'range': [None, 30]},
                       'bar': {'color': "orange"},
                       'steps': [{'range': [0, 10], 'color': "green"},
                                {'range': [10, 20], 'color': "yellow"},
                                {'range': [20, 30], 'color': "red"}]}
            ), row=2, col=2)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📈 Performance Analysis")
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Returns Distribution")
            
            # Generate mock returns data
            returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for a year
            
            fig = px.histogram(
                x=returns,
                nbins=30,
                title="Daily Returns Distribution",
                labels={'x': 'Daily Return', 'y': 'Frequency'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Cumulative Returns")
            
            # Generate cumulative returns
            cumulative_returns = (1 + pd.Series(returns)).cumprod()
            dates = pd.date_range(start='2023-01-01', periods=len(cumulative_returns), freq='D')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Returns Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        st.subheader("📉 Drawdown Analysis")
        
        # Calculate drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("💼 Current Positions")
        
        # Positions table
        positions_df = pd.DataFrame(portfolio_data['positions'])
        
        if not positions_df.empty:
            # Add color coding for P&L
            def color_pnl(val):
                if val > 0:
                    return 'color: green'
                elif val < 0:
                    return 'color: red'
                return ''
            
            # Format the dataframe
            display_df = positions_df.copy()
            display_df['P&L'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['P&L %'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['Entry Price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
            
            # Reorder columns
            display_df = display_df[['symbol', 'quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %']]
            display_df.columns = ['Symbol', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Position allocation pie chart
            st.subheader("📊 Position Allocation")
            
            fig = px.pie(
                positions_df,
                values='quantity',
                names='symbol',
                title="Position Allocation by Symbol"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")
    
    with tab4:
        st.header("📋 Trade History")
        
        # Trades table
        trades_df = pd.DataFrame(trades_data)
        
        if not trades_df.empty:
            # Format the dataframe
            display_trades = trades_df.copy()
            display_trades['Price'] = display_trades['price'].apply(lambda x: f"${x:.2f}")
            display_trades['P&L'] = display_trades['pnl'].apply(lambda x: f"${x:,.2f}" if x != 0 else "-")
            
            # Reorder columns
            display_trades = display_trades[['timestamp', 'symbol', 'side', 'quantity', 'Price', 'P&L', 'strategy']]
            display_trades.columns = ['Timestamp', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L', 'Strategy']
            
            st.dataframe(display_trades, use_container_width=True)
            
            # Trade analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Trade Side Distribution")
                side_counts = trades_df['side'].value_counts()
                fig = px.pie(values=side_counts.values, names=side_counts.index, title="Buy vs Sell Trades")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Trade Volume by Symbol")
                symbol_volume = trades_df.groupby('symbol')['quantity'].sum()
                fig = px.bar(x=symbol_volume.index, y=symbol_volume.values, title="Total Volume by Symbol")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades recorded")

if __name__ == "__main__":
    main() 