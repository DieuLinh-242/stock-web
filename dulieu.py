from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

CSV_FILE = BASE / "abc-17.csv"
OUT_DIR = BASE / "output"              
OUT_DIR.mkdir(exist_ok=True)

REF_DAY1 = 10000.0
LIMIT = 0.10
YEAR = 2025

C_REF = "#FFD400"
C_FLOOR = "#1E90FF"
C_CEIL = "#C45BFF"
C_UP = "#2EE62E"
C_DOWN = "#FF4F4F"
C_TXT = "#FFFFFF"

@dataclass
class StockData:
    prices: np.ndarray

    def __post_init__(self):
        P = self.prices
        self.n, self.k = P.shape
        self.days = np.arange(1, self.n + 1)
        self.dates = pd.date_range(f"{YEAR}-01-01", periods=self.n)

        self.open = P[:, 0]
        self.close = P[:, -1]
        self.high = P.max(axis=1)
        self.low = P.min(axis=1)

        self.ref = np.empty(self.n)
        self.ref[0] = REF_DAY1
        self.ref[1:] = self.close[:-1]
        self.floor = self.ref * (1 - LIMIT)
        self.ceil = self.ref * (1 + LIMIT)

        self.run_high = np.maximum.accumulate(self.high)
        self.run_low = np.minimum.accumulate(self.low)

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