"""
data_utils.py - Cấu hình chung + đọc/xử lý dữ liệu giá cổ phiếu.

File này KHÔNG vẽ gì cả. Cả video.py và analysis.py đều import từ đây,
nên muốn đổi file csv, màu sắc, giá tham chiếu... chỉ cần sửa ở đây.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- CẤU HÌNH
try:
    BASE = Path(__file__).resolve().parent
except NameError:                      # chạy trong notebook / Colab
    BASE = Path.cwd()

CSV_FILE = BASE / "abc-17.csv"
OUT_DIR = BASE / "output"              # ảnh + bảng của phần phân tích
OUT_DIR.mkdir(exist_ok=True)

REF_DAY1 = 10000.0                     # giá tham chiếu ngày đầu tiên
LIMIT = 0.10                           # biên độ sàn/trần: +-10%
YEAR = 2025                            # năm giả định để gán ngày/tháng/quý (365 ngày)

# Màu quy ước theo đề
C_REF = "#FFD400"                      # giá tham chiếu - vàng
C_FLOOR = "#1E90FF"                    # giá sàn - xanh dương
C_CEIL = "#C45BFF"                     # giá trần - tím
C_UP = "#2EE62E"                       # tăng - xanh lá
C_DOWN = "#FF4F4F"                     # giảm - đỏ
C_TXT = "#FFFFFF"


# ----------------------------------------------------------------- DỮ LIỆU
@dataclass
class StockData:
    """Giữ toàn bộ mảng giá và các đại lượng suy ra từ file csv."""
    prices: np.ndarray                 # shape (N ngày, K phiên)

    def __post_init__(self):
        P = self.prices
        self.n, self.k = P.shape
        self.days = np.arange(1, self.n + 1)
        self.dates = pd.date_range(f"{YEAR}-01-01", periods=self.n)

        self.open = P[:, 0]
        self.close = P[:, -1]
        self.high = P.max(axis=1)
        self.low = P.min(axis=1)

        # giá tham chiếu = đóng cửa hôm trước
        self.ref = np.empty(self.n)
        self.ref[0] = REF_DAY1
        self.ref[1:] = self.close[:-1]
        self.floor = self.ref * (1 - LIMIT)
        self.ceil = self.ref * (1 + LIMIT)

        # cao nhất / thấp nhất lũy kế
        self.run_high = np.maximum.accumulate(self.high)
        self.run_low = np.minimum.accumulate(self.low)

    # ---- dạng bảng pandas (dùng cho phần phân tích)
    def daily_frame(self) -> pd.DataFrame:
        """1 dòng = 1 ngày."""
        return pd.DataFrame({
            "day": self.days,
            "date": self.dates,
            "month": self.dates.month,
            "quarter": self.dates.quarter,
            "open": self.open, "close": self.close,
            "high": self.high, "low": self.low,
            "ref": self.ref, "floor": self.floor, "ceil": self.ceil,
            "range": self.high - self.low,
        })

    def long_frame(self) -> pd.DataFrame:
        """1 dòng = 1 mức giá (ngày x phiên) -> 365*7 dòng."""
        df = pd.DataFrame({
            "day": np.repeat(self.days, self.k),
            "session": np.tile(np.arange(1, self.k + 1), self.n),
            "price": self.prices.ravel(),
        })
        df["date"] = self.dates[df["day"].to_numpy() - 1]
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        return df


def load_stock_data(csv_file: Path = CSV_FILE) -> StockData:
    df = pd.read_csv(csv_file, header=None)
    return StockData(df.to_numpy(dtype=float))