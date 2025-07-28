from lightweight_charts import Chart
import yfinance as yf
from vnstock import Quote
import pandas as pd
from time import sleep

# df1 = yf.Ticker("AAPL").history(start="2025-01-01", end="2025-05-31", period="1d")
df1 = Quote(symbol="VCB").history(start="2024-01-01", end="2024-12-31", period="1d")
# .loc[:, ["Open", "High", "Low", "Close", "Volume"]].reset_index()
# df1.columns = df1.columns.str.lower()
# df1['date'] = df1['date'].dt.strftime('%Y-%m-%d')
# df1.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
# print(df1.head())

# df2 = yf.Ticker("AAPL").history(period="max").loc['2024', ["Open", "High", "Low", "Close", "Volume"]].reset_index()
df2 = Quote(symbol="VCB").history(start="2025-01-01", end="2025-05-31", period="1d")
# df2.columns = df2.columns.str.lower()
# df2['date'] = df2['date'].dt.strftime('%Y-%m-%d')
# df2.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

# print(df2.head())

if __name__ == '__main__':
    chart = Chart()

    # Columns: time | open | high | low | close | volume
    # df = pd.read_csv('ohlcv.csv')
    chart.set(df1)

    chart.show(block=False)

    last_close = df1.iloc[-1]['close']

    for i, series in df2.iterrows():
        chart.update(series)

        if series['close'] > 50 and last_close < 65:
            chart.marker(text='The price crossed $20!')

        last_close = series['close']
        sleep(0.1)
