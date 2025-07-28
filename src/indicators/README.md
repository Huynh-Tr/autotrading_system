# Technical Indicators Module

This module contains various technical indicators used in trading strategies. All indicators are implemented as pure functions that can be easily imported and used throughout the trading system.

## Available Indicators

### 1. Simple Moving Average (SMA)
**File:** `sma.py`

Functions:
- `calculate_sma(prices, window)` - Calculate simple moving average
- `calculate_sma_crossover(prices, short_window, long_window)` - Calculate SMA crossover components
- `calculate_sma_components(prices, short_window, long_window)` - Get all SMA components as dictionary

### 2. Relative Strength Index (RSI)
**File:** `rsi.py`

Functions:
- `calculate_rsi(prices, period=14)` - Calculate RSI indicator
- `calculate_rsi_components(prices, period=14, overbought_threshold=70, oversold_threshold=30)` - Get RSI with threshold lines

### 3. Moving Average Convergence Divergence (MACD)
**File:** `macd.py`

Functions:
- `calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)` - Calculate MACD components
- `calculate_macd_components(prices, fast_period=12, slow_period=26, signal_period=9)` - Get all MACD components as dictionary

### 4. Bollinger Bands
**File:** `bollinger_bands.py`

Functions:
- `calculate_bollinger_bands(prices, period=20, std_dev=2)` - Calculate Bollinger Bands
- `calculate_bollinger_components(prices, period=20, std_dev=2)` - Get all Bollinger Bands components as dictionary

## Usage Examples

```python
import pandas as pd
from src.indicators.sma import calculate_sma
from src.indicators.rsi import calculate_rsi
from src.indicators.macd import calculate_macd
from src.indicators.bollinger_bands import calculate_bollinger_bands

# Sample price data
prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

# Calculate indicators
sma_20 = calculate_sma(prices, 20)
rsi = calculate_rsi(prices, 14)
macd_line, signal_line, histogram = calculate_macd(prices)
upper_band, middle_band, lower_band = calculate_bollinger_bands(prices)
```

## Integration with Strategies

All trading strategies now use the indicators from this module instead of calculating them internally. This provides:

1. **Consistency** - All indicators are calculated the same way across the system
2. **Reusability** - Indicators can be used in multiple strategies and components
3. **Maintainability** - Changes to indicator calculations only need to be made in one place
4. **Testability** - Indicators can be tested independently

## Testing

Run the indicators test to verify all indicators are working correctly:

```bash
python test_indicators.py
```

## Adding New Indicators

To add a new indicator:

1. Create a new file in the `indicators` directory (e.g., `ema.py`)
2. Implement the indicator calculation functions
3. Add the import to `__init__.py`
4. Update this README with documentation
5. Add tests to `test_indicators.py` 