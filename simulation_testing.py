from lightweight_charts import Chart

import pandas as pd
# from vnstock import Quote

from src.data.data_manager import DataManager
from src.utils.config_manager import ConfigManager


class SimulationTesting:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = ConfigManager(config_path)
        self.data_manager = DataManager(self.config)

    def setup_logging(config: ConfigManager):
        """Setup logging configuration"""
        start_date = config.get("data.start_date")
        end_date = config.get("data.end_date")
        interval = config.get("data.interval")
        symbols = config.get("trading.symbols")
        print(f"Start date: {start_date}, End date: {end_date}, Interval: {interval}, Symbols: {symbols}")

    def run(self):
        # Get trades transactions
        trades_transactions = pd.read_csv('data/trades.csv')
        # start_date = trades_transactions['date'].min()
        # end_date = trades_transactions['date'].max()
        return trades_transactions

    # Get historical data
    def get_price_history(self, symbols, start_date: str, end_date: str, interval: str = "1d"):
        data = self.data_manager._fetch_vnstock_ohlcv_data(symbols, start_date, end_date, interval)
        return data

    # def plot_price_history(self, symbols: str, start_date: str, end_date: str):
    #     data = self.get_price_history(symbols, start_date, end_date)
    #     chart = Chart()
    #     chart.add_series(data)
    #     chart.show()

if __name__ == "__main__":
    simulation_testing = SimulationTesting()
    get_price_history = simulation_testing.get_price_history(
        symbols=simulation_testing.config.get("trading.symbols"), 
        start_date=simulation_testing.config.get("data.start_date"), 
        end_date=simulation_testing.config.get("data.end_date"), 
        interval=simulation_testing.config.get("data.interval")
    )[simulation_testing.config.get("trading.symbols")[0]]
    get_price_history.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    # get_price_history['date'] = pd.to_datetime(get_price_history['date'], format='%Y-%m-%d')

    # trades_transactions = simulation_testing.run()
    # print(get_price_history.date)

    chart = Chart()
    chart.set(get_price_history)
    chart.show(block=True)