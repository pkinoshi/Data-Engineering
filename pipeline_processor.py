"""
pipeline_processor.py
----------------------
Receives raw sensor records from the simulator (or a live stream),
cleans, engineers features, and prepares a model-ready DataFrame.

Design decisions aligned with study scope:
  - No external storage dependency (in-memory buffer mimics a stream)
  - Rolling window features capture temporal patterns
  - Normalisation is fit on a warm-up batch, then applied to each record
"""

import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = ["temperature", "vibration", "pressure", "rotational_speed"]
WINDOW_SIZE = 10          # rolling window for feature engineering


class StreamProcessor:
    """
    Processes incoming sensor records one-by-one (or in micro-batches).

    Usage
    -----
    processor = StreamProcessor()
    processor.fit_scaler(warmup_df)          # fit on first N rows
    processed_row = processor.process(row)   # returns feature dict or None
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._scaler = StandardScaler()
        self._scaler_fitted = False

    # ------------------------------------------------------------------
    # Scaler warm-up
    # ------------------------------------------------------------------

    def fit_scaler(self, warmup_df: pd.DataFrame) -> None:
        """Fit the scaler on a warm-up batch (e.g. first 100 rows)."""
        self._scaler.fit(warmup_df[SENSOR_COLS].values)
        self._scaler_fitted = True

    # ------------------------------------------------------------------
    # Per-record processing
    # ------------------------------------------------------------------

    def process(self, row: dict) -> dict | None:
        """
        Accept one raw sensor record and return an enriched feature dict.
        Returns None until the rolling window is sufficiently filled.
        """
        # 1. Basic validation / clipping (domain knowledge bounds)
        clean = self._validate(row)

        # 2. Add to rolling buffer
        self._buffer.append([clean[s] for s in SENSOR_COLS])

        if len(self._buffer) < self.window_size:
            return None   # not enough history yet

        # 3. Rolling statistics
        window_arr = np.array(self._buffer)
        rolling_means = window_arr.mean(axis=0)
        rolling_stds  = window_arr.std(axis=0)

        features = {"cycle":     clean["cycle"],
                    "timestamp": clean["timestamp"],
                    "is_fault":  clean["is_fault"]}

        for i, sensor in enumerate(SENSOR_COLS):
            features[sensor]                      = clean[sensor]
            features[f"{sensor}_roll_mean"]       = round(rolling_means[i], 4)
            features[f"{sensor}_roll_std"]        = round(rolling_stds[i],  4)

        # 4. Normalise raw sensor values (if scaler fitted)
        if self._scaler_fitted:
            raw_vals = np.array([[clean[s] for s in SENSOR_COLS]])
            scaled   = self._scaler.transform(raw_vals)[0]
            for i, sensor in enumerate(SENSOR_COLS):
                features[f"{sensor}_scaled"] = round(float(scaled[i]), 4)

        # 5. Pass-through prediction targets
        features["rul"]              = clean["rul"]
        features["health_index"]     = clean["health_index"]
        features["maintenance_flag"] = clean["maintenance_flag"]
        features["replace_flag"]     = clean["replace_flag"]

        return features

    # ------------------------------------------------------------------
    # Batch helper (process a whole DataFrame at once)
    # ------------------------------------------------------------------

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process every row in df and return cleaned feature DataFrame."""
        results = []
        for _, row in df.iterrows():
            out = self.process(row.to_dict())
            if out is not None:
                results.append(out)
        return pd.DataFrame(results).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, row: dict) -> dict:
        """Clip sensor readings to physically reasonable bounds."""
        clean = dict(row)
        bounds = {
            "temperature":      (40.0,  200.0),
            "vibration":        (0.0,     5.0),
            "pressure":         (50.0,  250.0),
            "rotational_speed": (500.0, 3000.0),
        }
        for sensor, (lo, hi) in bounds.items():
            clean[sensor] = float(np.clip(clean[sensor], lo, hi))
        return clean


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sensor_simulator import SensorSimulator

    sim = SensorSimulator(total_cycles=300)
    raw_df = sim.generate()

    proc = StreamProcessor(window_size=10)
    proc.fit_scaler(raw_df.head(100))          # warm-up scaler

    feature_df = proc.process_dataframe(raw_df)
    print(feature_df.head(5).to_string(index=False))
    print(f"\nFeature columns ({len(feature_df.columns)}): {list(feature_df.columns)}")
    print(f"Processed rows: {len(feature_df)}")
