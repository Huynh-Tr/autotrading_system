from TradingviewData import TradingViewData, Interval
from lightweight_charts import Chart
import pandas as pd

print("Initializing TradingViewData...")
request = TradingViewData()
print("Fetching data from TradingView...")
df = request.get_hist(symbol='VHC',exchange='HOSE',interval=Interval.daily,n_bars=10).drop(columns=['symbol']).reset_index(drop=False)
print("Data fetched successfully!")
print("Data shape:", df.shape)
print(df.head())

# chart = Chart().set(df)
# chart
# chart.show(block=True)