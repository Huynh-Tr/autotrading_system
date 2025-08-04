# Cell 1: Setup imports
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import TradingEngine và components
from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager
from src.data.data_manager import DataManager

# Import visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

print("Import thành công!")

# Cell 2: Khởi tạo TradingEngine và lấy dữ liệu
def initialize_system():
    """Khởi tạo hệ thống và lấy dữ liệu"""
    print("=" * 60)
    print("KHỞI TẠO HỆ THỐNG")
    print("=" * 60)
    
    # Khởi tạo TradingEngine
    config_path = "../../config/config.yaml"
    engine = TradingEngine(config_path)
    
    print(f"TradingEngine đã được khởi tạo với vốn ban đầu: ${engine.cash:,.2f}")
    print(f"Config loaded: {engine.config.get('trading.symbols')}")
    print(f"Config loaded: {engine.config.get('data.source')}")
    
    # Lấy dữ liệu
    symbols = engine.config.get("trading.symbols", ["AAPL"])
    start_date = engine.config.get("data.start_date", "2023-01-01")
    end_date = engine.config.get("data.end_date", "2023-12-31")
    
    # Lấy data từ DataManager
    data_manager = DataManager(engine.config)
    historical_data = data_manager.get_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=engine.config.get("data.interval", "1d"),
        n_bars=engine.config.get("data.n_bars", 1000)
    )
    
    print(f"Lấy data cho symbols: {symbols}")
    print(f"Data shape: {historical_data.shape}")
    print(f"Data columns: {historical_data.columns.tolist()}")
    print(f"Date range: {historical_data.index.min()} đến {historical_data.index.max()}")
    
    return engine, historical_data, symbols, start_date, end_date

# Cell 3: Hiển thị các chiến lược có sẵn
def show_available_strategies():
    """Hiển thị các chiến lược có sẵn"""
    print("=" * 60)
    print("CÁC CHIẾN LƯỢC CÓ SẴN")
    print("=" * 60)
    
    strategies = {
        'sma_crossover': {
            'name': 'SMA Crossover Strategy',
            'description': 'Chiến lược giao dịch dựa trên giao cắt của hai đường trung bình động',
            'parameters': ['short_window', 'long_window'],
            'default_params': {'short_window': 5, 'long_window': 30}
        },
        'rsi': {
            'name': 'RSI Strategy',
            'description': 'Chiến lược giao dịch dựa trên chỉ báo RSI (Relative Strength Index)',
            'parameters': ['period', 'oversold', 'overbought'],
            'default_params': {'period': 14, 'oversold': 30, 'overbought': 70}
        },
        'macd': {
            'name': 'MACD Strategy',
            'description': 'Chiến lược giao dịch dựa trên chỉ báo MACD (Moving Average Convergence Divergence)',
            'parameters': ['fast_period', 'slow_period', 'signal_period'],
            'default_params': {'fast_period': 15, 'slow_period': 20, 'signal_period': 7}
        }
    }
    
    for i, (strategy_id, strategy_info) in enumerate(strategies.items(), 1):
        print(f"{i}. {strategy_info['name']}")
        print(f"   Mô tả: {strategy_info['description']}")
        print(f"   Tham số: {', '.join(strategy_info['parameters'])}")
        print(f"   Tham số mặc định: {strategy_info['default_params']}")
        print()
    
    return strategies

# Cell 4: Người dùng chọn chiến lược
def select_strategies():
    """Cho phép người dùng chọn chiến lược"""
    print("=" * 60)
    print("CHỌN CHIẾN LƯỢC")
    print("=" * 60)
    
    strategies = show_available_strategies()
    
    print("Chọn chiến lược để tối ưu hóa (nhập số, cách nhau bởi dấu phẩy):")
    print("Ví dụ: 1,2,3 hoặc 1")
    
    while True:
        try:
            choice = input("Nhập lựa chọn của bạn: ").strip()
            if not choice:
                print("Vui lòng nhập lựa chọn!")
                continue
            
            # Parse choices
            choices = [int(x.strip()) for x in choice.split(',')]
            strategy_ids = list(strategies.keys())
            
            selected_strategies = []
            for choice_num in choices:
                if 1 <= choice_num <= len(strategy_ids):
                    strategy_id = strategy_ids[choice_num - 1]
                    selected_strategies.append(strategy_id)
                else:
                    print(f"Lựa chọn {choice_num} không hợp lệ!")
            
            if selected_strategies:
                print(f"\nĐã chọn các chiến lược: {selected_strategies}")
                return selected_strategies
            else:
                print("Không có chiến lược nào được chọn!")
                
        except ValueError:
            print("Vui lòng nhập số hợp lệ!")
        except Exception as e:
            print(f"Lỗi: {e}")

# Cell 5: Chạy optimization
def run_optimization(engine, selected_strategies, start_date, end_date):
    """Chạy optimization cho các chiến lược đã chọn"""
    print("=" * 60)
    print("CHẠY OPTIMIZATION")
    print("=" * 60)
    
    print(f"Tối ưu hóa các chiến lược: {selected_strategies}")
    print(f"Thời gian: {start_date} đến {end_date}")
    
    # Chạy optimization với data caching
    optimization_results = engine.optimize_strategies(
        strategy_types=selected_strategies,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=30,  # Giảm số lượng để demo nhanh
        symbols=engine.config.get("trading.symbols", ["AAPL"])
    )
    
    # Hiển thị kết quả optimization
    print("\nKẾT QUẢ OPTIMIZATION:")
    summary = engine.get_optimization_summary()
    
    for strategy_type, info in summary.items():
        print(f"\n{strategy_type.upper()}:")
        print(f"  Tham số tốt nhất: {info['best_parameters']}")
        print(f"  Sharpe Ratio: {info['best_metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"  Total Return: {info['best_metrics'].get('total_return', 0):.2%}")
        print(f"  Max Drawdown: {info['best_metrics'].get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {info['best_metrics'].get('win_rate', 0):.2%}")
    
    return optimization_results

# Cell 6: Chạy backtest với tham số đã tối ưu
def run_optimized_backtest(engine, selected_strategies, start_date, end_date):
    """Chạy backtest với tham số đã tối ưu"""
    print("=" * 60)
    print("CHẠY BACKTEST VỚI THAM SỐ ĐÃ TỐI ƯU")
    print("=" * 60)
    
    # Chạy optimized backtest với data caching
    backtest_results = engine.run_optimized_backtest(
        start_date=start_date,
        end_date=end_date,
        strategy_types=selected_strategies,
        symbols=engine.config.get("trading.symbols", ["AAPL"])
    )
    
    # Hiển thị kết quả backtest
    print("\nKẾT QUẢ BACKTEST:")
    for strategy_name, result in backtest_results.items():
        metrics = result.get('performance_metrics', {})
        print(f"\n{strategy_name}:")
        print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    return backtest_results

# Cell 6.5: Chạy complete workflow (mới)
def run_complete_workflow(engine, selected_strategies, start_date, end_date):
    """Chạy complete workflow với data caching"""
    print("=" * 60)
    print("CHẠY COMPLETE OPTIMIZATION WORKFLOW")
    print("=" * 60)
    
    print(f"Chiến lược: {selected_strategies}")
    print(f"Thời gian: {start_date} đến {end_date}")
    print("Sử dụng data caching để tối ưu hiệu suất...")
    
    # Chạy complete workflow với data caching
    complete_results = engine.run_complete_optimization_workflow(
        strategy_types=selected_strategies,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=30,
        symbols=engine.config.get("trading.symbols", ["AAPL"])
    )
    
    # Hiển thị thông tin data
    data_info = complete_results.get('data_info', {})
    print(f"\n=== THÔNG TIN DATA ===")
    print(f"Data shape: {data_info.get('data_shape')}")
    print(f"Symbols: {data_info.get('symbols')}")
    print(f"Period: {data_info.get('start_date')} đến {data_info.get('end_date')}")
    
    # Hiển thị kết quả optimization
    optimization_results = complete_results.get('optimization_results', {})
    print(f"\n=== KẾT QUẢ OPTIMIZATION ===")
    for strategy_type, result in optimization_results.items():
        if 'best_parameters' in result:
            best_params = result['best_parameters']
            print(f"\n{strategy_type.upper()}:")
            print(f"  Tham số tốt nhất: {best_params.get('parameters', {})}")
            metrics = best_params.get('metrics', {})
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    # Hiển thị kết quả backtest
    backtest_results = complete_results.get('backtest_results', {})
    print(f"\n=== KẾT QUẢ BACKTEST ===")
    for strategy_name, result in backtest_results.items():
        metrics = result.get('performance_metrics', {})
        print(f"\n{strategy_name}:")
        print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    return complete_results

# Cell 7: Visualization kết quả backtest
def visualize_backtest_results(backtest_results):
    """Tạo biểu đồ cho kết quả backtest"""
    print("=" * 60)
    print("VISUALIZATION KẾT QUẢ BACKTEST")
    print("=" * 60)
    
    if not backtest_results:
        print("Không có kết quả backtest để visualize!")
        return
    
    # Chuẩn bị dữ liệu cho visualization
    strategy_names = list(backtest_results.keys())
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    # Tạo subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_names,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        values = []
        for strategy_name in strategy_names:
            result = backtest_results[strategy_name]
            performance_metrics = result.get('performance_metrics', {})
            value = performance_metrics.get(metric, 0)
            values.append(value)
        
        # Tạo bar chart
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=values,
                name=metric_name,
                marker_color=colors[i % len(colors)]
            ),
            row=row, col=col
        )
        
        # Cập nhật layout cho từng subplot
        fig.update_xaxes(title_text="Strategy", row=row, col=col)
        fig.update_yaxes(title_text=metric_name, row=row, col=col)
    
    fig.update_layout(
        height=800,
        title_text="Backtest Results Comparison",
        showlegend=False
    )
    
    fig.show()
    
    # Tạo heatmap cho correlation giữa các metrics
    create_metrics_heatmap(backtest_results)

def create_metrics_heatmap(backtest_results):
    """Tạo heatmap cho correlation giữa các metrics"""
    if not backtest_results:
        return
    
    # Chuẩn bị dữ liệu
    metrics_data = []
    strategy_names = []
    
    for strategy_name, result in backtest_results.items():
        performance_metrics = result.get('performance_metrics', {})
        strategy_names.append(strategy_name)
        
        metrics_data.append([
            performance_metrics.get('total_return', 0),
            performance_metrics.get('sharpe_ratio', 0),
            performance_metrics.get('max_drawdown', 0),
            performance_metrics.get('win_rate', 0),
            performance_metrics.get('profit_factor', 0)
        ])
    
    # Tạo DataFrame
    metrics_df = pd.DataFrame(
        metrics_data,
        index=strategy_names,
        columns=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    )
    
    # Tạo heatmap
    fig = go.Figure(data=go.Heatmap(
        z=metrics_df.values,
        x=metrics_df.columns,
        y=metrics_df.index,
        colorscale='RdYlBu',
        text=metrics_df.values.round(3),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Performance Metrics Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Strategies",
        height=500
    )
    
    fig.show()

# Cell 8: Hiển thị chi tiết trades
def show_trade_details(engine):
    """Hiển thị chi tiết trades"""
    print("=" * 60)
    print("CHI TIẾT TRADES")
    print("=" * 60)
    
    if hasattr(engine, 'trades') and engine.trades:
        print(f"Tổng số trades: {len(engine.trades)}")
        
        # Tạo DataFrame cho trades
        trades_df = pd.DataFrame([
            {
                'Symbol': trade.symbol,
                'Side': trade.side,
                'Quantity': trade.quantity,
                'Price': trade.price,
                'Timestamp': trade.timestamp,
                'Commission': trade.commission,
                'Strategy': trade.strategy,
                'PnL': getattr(trade, 'pnl', 0),
                'PnL_Pct': getattr(trade, 'pnl_pct', 0)
            }
            for trade in engine.trades
        ])
        
        print("\nTrades DataFrame:")
        print(trades_df.head(10))
        
        # Thống kê trades
        print(f"\n=== THỐNG KÊ TRADES ===")
        print(f"Total Trades: {len(engine.trades)}")
        print(f"Buy Trades: {len(trades_df[trades_df['Side'] == 'buy'])}")
        print(f"Sell Trades: {len(trades_df[trades_df['Side'] == 'sell'])}")
        print(f"Average Trade Size: {trades_df['Quantity'].mean():.2f}")
        print(f"Total Commission: ${trades_df['Commission'].sum():.2f}")
        
        # Thống kê theo strategy
        if 'Strategy' in trades_df.columns:
            print(f"\n=== TRADES THEO CHIẾN LƯỢC ===")
            strategy_stats = trades_df['Strategy'].value_counts()
            for strategy, count in strategy_stats.items():
                print(f"{strategy}: {count} trades")
        
        return trades_df
    else:
        print("Không có trades nào được thực hiện")
        return None

# Cell 9: Visualization trades
def visualize_trades(trades_df, historical_data, symbols):
    """Visualization trades bằng hàm viz"""
    print("=" * 60)
    print("VISUALIZATION TRADES")
    print("=" * 60)
    
    if trades_df is None or trades_df.empty:
        print("Không có trades để visualize!")
        return
    
    # Sử dụng symbol đầu tiên
    symbol = symbols[0] if symbols else list(historical_data.columns)[0]
    
    print(f"Visualizing trades cho symbol: {symbol}")
    
    # Gọi hàm viz
    viz(symbol, trades_df, historical_data)

def viz(symbol, trades_df, historical_data):
    """Visualize trades trên biểu đồ candlestick"""
    # Tạo subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.8, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_data[symbol].index,
            open=historical_data[symbol].open,
            high=historical_data[symbol].high,
            low=historical_data[symbol].low,
            close=historical_data[symbol].close,
            name='Price'
        ), 
        row=1, col=1
    )
    
    # Set y-axis limits
    set_ylim = (
        historical_data[symbol].low.min() * 0.98, 
        historical_data[symbol].high.max() * 1.02
    )
    
    # Add buy/sell markers
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['Side'] == 'buy']
        sell_trades = trades_df[trades_df['Side'] == 'sell']
        
        # Buy markers
        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_trades['Timestamp'],
                    y=historical_data[symbol][historical_data.index.isin(buy_trades['Timestamp'])].low * 0.998,
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', color='green', size=12),
                    text=['Buy']*len(buy_trades),
                    textposition='bottom center',
                    name='Buy'
                ),
                row=1, col=1
            )
        
        # Sell markers
        if not sell_trades.empty():
            fig.add_trace(
                go.Scatter(
                    x=sell_trades['Timestamp'],
                    y=historical_data[symbol][historical_data.index.isin(sell_trades['Timestamp'])].high * 1.002,
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    text=['Sell']*len(sell_trades),
                    textposition='top center',
                    name='Sell'
                ),
                row=1, col=1
            )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=historical_data[symbol].index,
            y=historical_data[symbol].volume,
            name='Volume'
        ), 
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'Trading Activity - {symbol}',
        yaxis_range=(set_ylim[0], set_ylim[1]),
        xaxis_title='Date',
        yaxis_title='Price', 
        height=800, 
        width=1000
    )
    
    fig.show()

# Cell 10: Main workflow
def main_workflow():
    """Workflow chính"""
    print("=" * 80)
    print("TRADING SYSTEM WORKFLOW")
    print("=" * 80)
    
    # Bước 1: Khởi tạo hệ thống
    engine, historical_data, symbols, start_date, end_date = initialize_system()
    
    # Bước 2: Chọn chiến lược
    selected_strategies = select_strategies()
    
    # Bước 3: Chạy complete workflow (optimization + backtest với data caching)
    complete_results = run_complete_workflow(engine, selected_strategies, start_date, end_date)
    
    # Bước 4: Visualization kết quả backtest
    backtest_results = complete_results.get('backtest_results', {})
    visualize_backtest_results(backtest_results)
    
    # Bước 5: Hiển thị chi tiết trades
    trades_df = show_trade_details(engine)
    
    # Bước 6: Visualization trades
    visualize_trades(trades_df, historical_data, symbols)
    
    # Bước 7: Hiển thị thông tin data caching
    data_info = engine.get_cached_data_info()
    print("\n" + "=" * 60)
    print("THÔNG TIN DATA CACHING")
    print("=" * 60)
    print(f"Có cached data: {data_info['has_cached_data']}")
    print(f"Data shape: {data_info['data_shape']}")
    print(f"Start date: {data_info['start_date']}")
    print(f"End date: {data_info['end_date']}")
    print(f"Symbols: {data_info['symbols']}")
    
    print("\n" + "=" * 80)
    print("WORKFLOW HOÀN THÀNH!")
    print("=" * 80)

# Chạy workflow
if __name__ == "__main__":
    main_workflow()