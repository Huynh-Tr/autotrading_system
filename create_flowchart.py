#!/usr/bin/env python3
"""
Create Visual Flowchart for Auto Trading System
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_flowchart():
    """Create a comprehensive flowchart of the auto trading system"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'config': '#e1f5fe',
        'data': '#f3e5f5', 
        'strategy': '#e8f5e8',
        'risk': '#fff3e0',
        'execution': '#fce4ec',
        'performance': '#f1f8e9',
        'dashboard': '#e0f2f1',
        'storage': '#fafafa'
    }
    
    # Define box positions and sizes
    boxes = {
        # Configuration & Initialization
        'start': {'pos': (5, 9.5), 'size': (2, 0.4), 'text': 'User Starts System', 'color': colors['config']},
        'config': {'pos': (5, 8.8), 'size': (2, 0.4), 'text': 'Load Configuration\nconfig/config.yaml', 'color': colors['config']},
        'engine': {'pos': (5, 8.1), 'size': (2, 0.4), 'text': 'Initialize Trading Engine\nsrc/core/trading_engine.py', 'color': colors['config']},
        
        # Data Management
        'data_manager': {'pos': (2, 7.4), 'size': (2, 0.4), 'text': 'Data Manager\nsrc/data/data_manager.py', 'color': colors['data']},
        'fetch_data': {'pos': (2, 6.7), 'size': (2, 0.4), 'text': 'Fetch Market Data\nYahoo Finance API', 'color': colors['data']},
        'validate_data': {'pos': (2, 6.0), 'size': (2, 0.4), 'text': 'Data Validation & Cleaning', 'color': colors['data']},
        'indicators': {'pos': (2, 5.3), 'size': (2, 0.4), 'text': 'Calculate Technical Indicators\nSMA, RSI, MACD, BB', 'color': colors['data']},
        'cache': {'pos': (2, 4.6), 'size': (2, 0.4), 'text': 'Cache Data\ndata/cache/', 'color': colors['storage']},
        
        # Strategy Processing
        'strategy': {'pos': (8, 7.4), 'size': (2, 0.4), 'text': 'Strategy Selection\nSMA Crossover, RSI, MACD', 'color': colors['strategy']},
        'signals': {'pos': (8, 6.7), 'size': (2, 0.4), 'text': 'Generate Trading Signals\nsrc/strategies/sma_crossover.py', 'color': colors['strategy']},
        'signal_type': {'pos': (8, 6.0), 'size': (2, 0.4), 'text': 'Signal Type?\nBuy/Sell/Hold', 'color': colors['strategy']},
        
        # Risk Management
        'risk_manager': {'pos': (5, 5.3), 'size': (2, 0.4), 'text': 'Risk Management\nsrc/risk/risk_manager.py', 'color': colors['risk']},
        'risk_check': {'pos': (5, 4.6), 'size': (2, 0.4), 'text': 'Risk Check\nPosition Size, Portfolio Risk', 'color': colors['risk']},
        
        # Order Execution
        'buy_order': {'pos': (2, 3.9), 'size': (2, 0.4), 'text': 'Execute Buy Order\nsrc/core/trading_engine.py', 'color': colors['execution']},
        'sell_order': {'pos': (8, 3.9), 'size': (2, 0.4), 'text': 'Execute Sell Order\nsrc/core/trading_engine.py', 'color': colors['execution']},
        'update_portfolio': {'pos': (5, 3.2), 'size': (2, 0.4), 'text': 'Update Portfolio\nCash, Positions, P&L', 'color': colors['execution']},
        'record_trade': {'pos': (5, 2.5), 'size': (2, 0.4), 'text': 'Record Trade\nTrade History', 'color': colors['execution']},
        
        # Performance Tracking
        'performance': {'pos': (5, 1.8), 'size': (2, 0.4), 'text': 'Performance Metrics\nReturns, Sharpe, Drawdown', 'color': colors['performance']},
        'risk_metrics': {'pos': (5, 1.1), 'size': (2, 0.4), 'text': 'Risk Metrics\nVolatility, Win Rate', 'color': colors['performance']},
        
        # Dashboard
        'dashboard': {'pos': (8, 1.8), 'size': (2, 0.4), 'text': 'Streamlit Dashboard\ndashboard/streamlit_app.py', 'color': colors['dashboard']},
        'portfolio_view': {'pos': (8, 1.1), 'size': (2, 0.4), 'text': 'Portfolio Overview\nValue, P&L, Positions', 'color': colors['dashboard']},
        
        # Storage & Logging
        'logging': {'pos': (2, 1.8), 'size': (2, 0.4), 'text': 'Logging\nlogs/trading.log', 'color': colors['storage']},
        'storage': {'pos': (2, 1.1), 'size': (2, 0.4), 'text': 'Data Storage\nCSV, SQLite, Cache', 'color': colors['storage']},
        
        # Control Flow
        'continue': {'pos': (5, 0.4), 'size': (2, 0.4), 'text': 'Continue Trading?\nLoop Back to Data Fetch', 'color': colors['config']}
    }
    
    # Draw boxes
    for name, box in boxes.items():
        x, y = box['pos']
        width, height = box['size']
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, box['text'], ha='center', va='center', fontsize=8, 
                fontweight='bold', wrap=True)
    
    # Draw arrows
    arrows = [
        # Configuration flow
        ('start', 'config'),
        ('config', 'engine'),
        ('engine', 'data_manager'),
        ('engine', 'strategy'),
        
        # Data flow
        ('data_manager', 'fetch_data'),
        ('fetch_data', 'validate_data'),
        ('validate_data', 'indicators'),
        ('indicators', 'cache'),
        
        # Strategy flow
        ('strategy', 'signals'),
        ('signals', 'signal_type'),
        ('signal_type', 'risk_manager'),
        ('indicators', 'signals'),
        
        # Risk management
        ('risk_manager', 'risk_check'),
        ('risk_check', 'buy_order'),
        ('risk_check', 'sell_order'),
        
        # Order execution
        ('buy_order', 'update_portfolio'),
        ('sell_order', 'update_portfolio'),
        ('update_portfolio', 'record_trade'),
        
        # Performance tracking
        ('record_trade', 'performance'),
        ('performance', 'risk_metrics'),
        ('risk_metrics', 'dashboard'),
        ('risk_metrics', 'portfolio_view'),
        
        # Storage
        ('record_trade', 'logging'),
        ('performance', 'storage'),
        
        # Control flow
        ('portfolio_view', 'continue'),
        ('continue', 'fetch_data'),
        
        # Cross connections
        ('cache', 'signals'),
        ('logging', 'performance'),
        ('storage', 'dashboard')
    ]
    
    # Draw arrows
    for start, end in arrows:
        start_pos = boxes[start]['pos']
        end_pos = boxes[end]['pos']
        
        # Calculate arrow direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Adjust for box sizes
        start_height = boxes[start]['size'][1]
        end_height = boxes[end]['size'][1]
        
        if dy > 0:  # Upward arrow
            start_y = start_pos[1] + start_height/2
            end_y = end_pos[1] - end_height/2
        else:  # Downward arrow
            start_y = start_pos[1] - start_height/2
            end_y = end_pos[1] + end_height/2
        
        # Draw arrow
        arrow = ConnectionPatch(
            (start_pos[0], start_y), (end_pos[0], end_y),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="black", ec="black",
            linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 9.8, 'Auto Trading System - Complete Data Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['config'], label='Configuration'),
        patches.Patch(color=colors['data'], label='Data Management'),
        patches.Patch(color=colors['strategy'], label='Strategy Processing'),
        patches.Patch(color=colors['risk'], label='Risk Management'),
        patches.Patch(color=colors['execution'], label='Order Execution'),
        patches.Patch(color=colors['performance'], label='Performance Tracking'),
        patches.Patch(color=colors['dashboard'], label='Dashboard & Monitoring'),
        patches.Patch(color=colors['storage'], label='Storage & Logging')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.9))
    
    # Save the flowchart
    plt.tight_layout()
    plt.savefig('docs/flowchart.png', dpi=300, bbox_inches='tight')
    plt.savefig('docs/flowchart.pdf', bbox_inches='tight')
    
    print("✅ Flowchart created successfully!")
    print("📁 Files saved:")
    print("   - docs/flowchart.png (High resolution image)")
    print("   - docs/flowchart.pdf (Vector format)")
    
    plt.show()

def create_simplified_flowchart():
    """Create a simplified flowchart focusing on the main data flow"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Simplified flow
    steps = [
        {'pos': (5, 9), 'text': '1. Fetch Market Data\n(Yahoo Finance)', 'color': '#e3f2fd'},
        {'pos': (5, 8), 'text': '2. Calculate Indicators\n(SMA, RSI, MACD)', 'color': '#f3e5f5'},
        {'pos': (5, 7), 'text': '3. Generate Signals\n(Buy/Sell/Hold)', 'color': '#e8f5e8'},
        {'pos': (5, 6), 'text': '4. Risk Management\n(Position Sizing)', 'color': '#fff3e0'},
        {'pos': (5, 5), 'text': '5. Execute Orders\n(Buy/Sell)', 'color': '#fce4ec'},
        {'pos': (5, 4), 'text': '6. Update Portfolio\n(Cash, Positions)', 'color': '#f1f8e9'},
        {'pos': (5, 3), 'text': '7. Calculate Performance\n(Returns, Metrics)', 'color': '#e0f2f1'},
        {'pos': (5, 2), 'text': '8. Update Dashboard\n(Real-time Display)', 'color': '#fafafa'},
        {'pos': (5, 1), 'text': '9. Store Data\n(Logs, Database)', 'color': '#f5f5f5'}
    ]
    
    # Draw steps
    for i, step in enumerate(steps):
        x, y = step['pos']
        
        # Draw box
        rect = FancyBboxPatch(
            (x - 1.5, y - 0.3), 3, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=step['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, step['text'], ha='center', va='center', fontsize=10, 
                fontweight='bold')
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            next_y = steps[i + 1]['pos'][1]
            arrow = ConnectionPatch(
                (x, y - 0.3), (x, next_y + 0.3),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="black", ec="black",
                linewidth=2
            )
            ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 9.5, 'Auto Trading System - Simplified Data Flow', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add loop arrow
    loop_arrow = ConnectionPatch(
        (5, 1 - 0.3), (5, 9 + 0.3),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=20, fc="red", ec="red",
        linewidth=2, linestyle='--'
    )
    ax.add_patch(loop_arrow)
    
    # Add loop label
    ax.text(6.5, 5, 'Continuous Loop\n(Real-time Trading)', 
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/simplified_flowchart.png', dpi=300, bbox_inches='tight')
    print("✅ Simplified flowchart created: docs/simplified_flowchart.png")
    plt.show()

if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    import os
    os.makedirs('docs', exist_ok=True)
    
    print("Creating Auto Trading System Flowcharts...")
    print("=" * 50)
    
    # Create both flowcharts
    create_flowchart()
    create_simplified_flowchart()
    
    print("\n🎉 Flowchart generation complete!")
    print("\nFiles created:")
    print("📊 docs/flowchart.md - Detailed Mermaid flowchart")
    print("🖼️  docs/flowchart.png - Complete system flowchart")
    print("📄 docs/flowchart.pdf - Vector format flowchart")
    print("🖼️  docs/simplified_flowchart.png - Simplified flow") 