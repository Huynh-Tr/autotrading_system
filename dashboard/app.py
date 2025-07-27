"""
Trading Dashboard - Web interface for monitoring the trading system
"""

from flask import Flask, render_template, jsonify, request
import json
import os
import sys
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config_manager import ConfigManager

app = Flask(__name__)

# Load configuration
config = ConfigManager("config/config.yaml")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio summary"""
    # This would normally come from the trading engine
    # For now, return mock data
    portfolio_data = {
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
            }
        ]
    }
    return jsonify(portfolio_data)

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    # Mock performance data
    performance_data = {
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'volatility': 0.15,
        'win_rate': 0.65,
        'total_trades': 25
    }
    return jsonify(performance_data)

@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    positions = [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.00,
            'current_price': 155.00,
            'pnl': 500,
            'pnl_pct': 3.33,
            'strategy': 'SMA Crossover'
        },
        {
            'symbol': 'GOOGL',
            'quantity': 50,
            'entry_price': 2800.00,
            'current_price': 2850.00,
            'pnl': 2500,
            'pnl_pct': 1.79,
            'strategy': 'SMA Crossover'
        }
    ]
    return jsonify(positions)

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    trades = [
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
        }
    ]
    return jsonify(trades)

@app.route('/api/system_status')
def get_system_status():
    """Get system status"""
    status = {
        'status': 'running',
        'mode': 'backtest',
        'strategy': 'SMA Crossover',
        'last_update': datetime.now().isoformat(),
        'uptime': '2 hours 15 minutes'
    }
    return jsonify(status)

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify(config.get_all())

if __name__ == '__main__':
    host = config.get("dashboard.host", "0.0.0.0")
    port = config.get("dashboard.port", 5000)
    debug = config.get("dashboard.debug", False)
    
    print(f"Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug) 