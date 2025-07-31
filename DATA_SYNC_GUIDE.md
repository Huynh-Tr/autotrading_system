# Hướng dẫn Đồng bộ hóa DataFrame từ các nguồn dữ liệu

## Tổng quan

Hệ thống đồng bộ hóa dữ liệu được thiết kế để chuẩn hóa DataFrame output từ các nguồn dữ liệu khác nhau (Yahoo Finance, VNStock, TradingView) thành một format thống nhất.

## Cấu trúc hệ thống

```
src/data/
├── data_manager.py          # DataManager với tích hợp đồng bộ hóa
└── config/config.yaml       # Cấu hình hệ thống
```

## Các nguồn dữ liệu được hỗ trợ

### 1. Yahoo Finance (yfinance)
- **Format gốc**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **MultiIndex**: Có thể có cấu trúc MultiIndex cho nhiều symbols
- **Đặc điểm**: Dữ liệu quốc tế, đầy đủ OHLCV

### 2. VNStock (vnstock)
- **Format gốc**: `time`, `open`, `high`, `low`, `close`, `volume`
- **Đặc điểm**: Dữ liệu thị trường Việt Nam, đơn giản

### 3. TradingView (tradingview)
- **Format gốc**: `datetime`, `open`, `high`, `low`, `close`, `volume`
- **Đặc điểm**: Dữ liệu real-time, nhiều timeframes

## Format chuẩn hóa

Tất cả dữ liệu sẽ được chuẩn hóa thành format:

```python
STANDARD_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']
STANDARD_DTYPES = {
    'time': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64', 
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64'
}
```

## Cách sử dụng

### Sử dụng DataManager (tự động đồng bộ hóa)

```python
from src.data.data_manager import DataManager
from src.utils.config_manager import ConfigManager

# Khởi tạo DataManager
config = ConfigManager("config/config.yaml")
data_manager = DataManager(config)

# Lấy dữ liệu (sẽ được đồng bộ hóa tự động)
data = data_manager.get_historical_data(
    symbols=['AAPL', 'VNM'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='1d'
)
```

## Tính năng chính

### 1. Chuẩn hóa dữ liệu
- **Tên cột**: Chuẩn hóa tên cột thành format thống nhất
- **Kiểu dữ liệu**: Đảm bảo kiểu dữ liệu đúng (datetime, float64)
- **Cấu trúc**: Đảm bảo có đủ các cột cần thiết

### 2. Validation dữ liệu
- **Cấu trúc**: Kiểm tra đủ các cột cần thiết
- **Kiểu dữ liệu**: Kiểm tra kiểu dữ liệu đúng
- **Logic OHLC**: Kiểm tra tính logic của dữ liệu OHLC
- **Giá trị âm**: Phát hiện và cảnh báo giá trị âm

### 3. Làm sạch dữ liệu
- **Missing values**: Forward fill và loại bỏ NaN
- **Giá trị âm**: Chuyển đổi thành 0
- **Logic OHLC**: Sửa các lỗi logic (high < low, etc.)
- **Outliers**: Xử lý các giá trị bất thường

### 4. Báo cáo tóm tắt
- **Thống kê**: Số lượng records, date range
- **Missing values**: Thống kê missing values
- **Data quality**: Đánh giá chất lượng dữ liệu
- **Sources**: Thông tin về nguồn dữ liệu

## Ví dụ sử dụng

### Ví dụ 1: Lấy dữ liệu từ Yahoo Finance

```python
from src.data.data_manager import DataManager
from src.utils.config_manager import ConfigManager

# Khởi tạo
config = ConfigManager("config/config.yaml")
data_manager = DataManager(config)

# Lấy dữ liệu
data = data_manager.get_historical_data(
    symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='1d'
)

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
```

### Ví dụ 2: Validate và clean dữ liệu

```python
# Validate dữ liệu
is_valid = data_manager.validate_ohlcv_data(data)
print(f"Data validation: {'PASS' if is_valid else 'FAIL'}")

# Clean dữ liệu nếu cần
if not is_valid:
    cleaned_data = data_manager.clean_ohlcv_data(data)
    print("Data cleaned successfully")
```

### Ví dụ 3: Lấy dữ liệu từ VNStock

```python
# Cập nhật config để sử dụng VNStock
config.set("data.source", "vnstock")

# Lấy dữ liệu VNStock
vn_data = data_manager.get_historical_data(
    symbols=['VNM'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='1d'
)
```

## Cấu hình

### Config cho DataManager

```yaml
# config/config.yaml
data:
  source: "yfinance"  # yfinance, vnstock, tradingview
  cache_data: true
  cache_dir: "data/cache"
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  interval: "1d"
```

## Troubleshooting

### Lỗi Import
```python
# Kiểm tra imports
try:
    from src.data.data_manager import DataManager
    print("✅ DataManager imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### Lỗi Data Source
```python
# Kiểm tra data source
supported_sources = ['yfinance', 'vnstock', 'tradingview']
if source not in supported_sources:
    print(f"❌ Unsupported data source: {source}")
```

### Lỗi Validation
```python
# Kiểm tra validation
is_valid = data_manager.validate_ohlcv_data(data)
if not is_valid:
    print("❌ Data validation failed")
    # Clean data
    cleaned_data = data_manager.clean_ohlcv_data(data)
```

## Lưu ý quan trọng

1. **Data Source**: Đảm bảo data source được hỗ trợ
2. **Dependencies**: Cài đặt tất cả required packages
3. **Cache**: Dữ liệu đã đồng bộ hóa sẽ được cache
4. **Validation**: Luôn validate dữ liệu trước khi sử dụng
5. **Error Handling**: Xử lý lỗi khi dữ liệu không hợp lệ

## Performance

- **Caching**: Dữ liệu đã đồng bộ hóa được cache để tăng tốc
- **Validation**: Chỉ validate khi cần thiết
- **Memory**: Tối ưu hóa memory usage cho large datasets
- **Parallel**: Có thể mở rộng để xử lý song song

## Mở rộng

### Thêm nguồn dữ liệu mới

1. Thêm method `_fetch_newsource_ohlcv_data()` trong `DataManager`
2. Cập nhật logic trong `get_historical_data()`
3. Test với dữ liệu mẫu

### Tùy chỉnh validation rules

```python
# Tùy chỉnh validation rules trong DataManager
def custom_validation_rules(self, df):
    # Thêm rules tùy chỉnh
    pass
```

Hệ thống đồng bộ hóa dữ liệu này đảm bảo tính nhất quán và độ tin cậy của dữ liệu từ các nguồn khác nhau, giúp hệ thống auto trading hoạt động ổn định và chính xác. 