import plotly.graph_objects as go
from plotly.subplots import make_subplots

def viz(symbol, trades_df, historical_data):
    # Visualize
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=historical_data[symbol].index,
                    open=historical_data[symbol].open,
                    high=historical_data[symbol].high,
                    low=historical_data[symbol].low,
                    close=historical_data[symbol].close,
                    ), row=1, col=1)
    set_ylim = (historical_data[symbol].low.min() * 0.98, historical_data[symbol].high.max() * 1.02)
    # Add buy/sell markers from trades_df to the candlestick chart (row 1)
    if 'trades_df' in locals():
        buy_trades = trades_df[trades_df['Side'] == 'buy']
        sell_trades = trades_df[trades_df['Side'] == 'sell']
        # Buy markers
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

    fig.add_trace(go.Bar(x=historical_data[symbol].index,
                        y=historical_data[symbol].volume,
                        ), row=2, col=1)

    fig.update_layout(title=f'{symbol}',
                    yaxis_range=(set_ylim[0], set_ylim[1]),
                    xaxis_title='Date',
                    yaxis_title='Price', 
                    height=800, width=1000)
    fig.show()