"""
predictive_models.py
--------------------
Two lightweight, pipeline-friendly models:

  1. RULRegressor   - Linear Regression → estimates Remaining Useful Life (cycles)
  2. MaintenanceClassifier - Decision Tree → predicts maintenance flag
     ("Normal", "Warning", "Maintenance Required")

Both models are justified in the study (Section 3.4) by:
  simplicity · speed · interpretability · ease of integration
"""

import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    mean_squared_error, root_mean_squared_error,
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

SENSOR_COLS   = ["temperature", "vibration", "pressure", "rotational_speed"]
ROLL_MEAN_COLS = [f"{s}_roll_mean" for s in SENSOR_COLS]
ROLL_STD_COLS  = [f"{s}_roll_std"  for s in SENSOR_COLS]
SCALED_COLS    = [f"{s}_scaled"    for s in SENSOR_COLS]

# Features available after StreamProcessor
FEATURE_COLS = (SENSOR_COLS + ROLL_MEAN_COLS + ROLL_STD_COLS +
                ["cycle", "health_index"])


# ---------------------------------------------------------------------------
# RUL Regression
# ---------------------------------------------------------------------------

class RULRegressor:
    """Estimates Remaining Useful Life using Linear Regression."""

    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self._metrics: dict = {}

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        X = np.array(df[FEATURE_COLS].values, dtype=float)
        y = np.array(df["rul"].values, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=False
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        self._metrics = {
            "MSE":  round(mse,  4),
            "RMSE": round(rmse, 4),
            "test_samples": len(y_test),
        }
        return self._metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return RUL predictions (clipped to ≥ 0)."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.model.predict(np.array(df[FEATURE_COLS].values, dtype=float))
        return np.clip(preds, 0, None).round(1)

    @property
    def metrics(self) -> dict:
        return self._metrics


# ---------------------------------------------------------------------------
# Maintenance Classifier
# ---------------------------------------------------------------------------

class MaintenanceClassifier:
    """Classifies system condition using a Decision Tree."""

    LABELS = ["Normal", "Warning", "Maintenance Required"]

    def __init__(self, max_depth: int = 5):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, random_state=42
        )
        self.trained = False
        self._metrics: dict = {}

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        X = np.array(df[FEATURE_COLS].values, dtype=float)
        y = np.array(df["maintenance_flag"].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=False
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)

        self._metrics = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "test_samples": len(y_test),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=self.LABELS
            ).tolist(),
        }
        return self._metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(np.array(df[FEATURE_COLS].values, dtype=float))

    @property
    def metrics(self) -> dict:
        return self._metrics


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sensor_simulator  import SensorSimulator
    from pipeline_processor import StreamProcessor

    sim    = SensorSimulator(total_cycles=1000)
    raw_df = sim.generate()

    proc = StreamProcessor()
    proc.fit_scaler(raw_df.head(100))
    feat_df = proc.process_dataframe(raw_df)

    print(f"Feature rows available: {len(feat_df)}")

    rul_model = RULRegressor()
    reg_metrics = rul_model.fit(feat_df)
    print(f"\n[RUL Regressor] {reg_metrics}")

    cls_model = MaintenanceClassifier()
    cls_metrics = cls_model.fit(feat_df)
    print(f"\n[Maintenance Classifier] Accuracy={cls_metrics['Accuracy']}  "
          f"Precision={cls_metrics['Precision']}  Recall={cls_metrics['Recall']}")
    print("\nClassification Report:\n", cls_metrics["classification_report"])
