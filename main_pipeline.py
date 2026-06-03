"""
main_pipeline.py
----------------
Orchestrates the full Real-Time IoT Data Pipeline:

  Stage 1 → Simulate sensor data
  Stage 2 → Stream-process & feature-engineer
  Stage 3 → Train predictive models
  Stage 4 → Predict RUL and maintenance condition
  Stage 5 → Decision support
  Stage 6 → Write all outputs to Google Sheets

Configuration
-------------
Set GOOGLE_SHEETS_ENABLED = True and fill in CREDS_FILE / SHEET_ID to write
to Google Sheets.  When disabled, the pipeline falls back to CSV so you can
develop and test without credentials.

Run:
    python main_pipeline.py
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ── project modules
from sensor_simulator    import SensorSimulator
from pipeline_processor  import StreamProcessor
from predictive_models   import RULRegressor, MaintenanceClassifier
from decision_support    import evaluate_batch

# ── Google Sheets configuration
GOOGLE_SHEETS_ENABLED = True          # ← set False to fall back to CSV
CREDS_FILE            = "credentials.json"
SHEET_ID              = "1SO61B4KSnWAqKYUG3LocwspGEjZbR2_dF1gGue3S-K0"   # ← paste your spreadsheet ID here

# ── fallback CSV output directory (used when Sheets is disabled)
OUTPUT_DIR = "pipeline_outputs"

# ── pipeline configuration
CONFIG = {
    "total_cycles":      2000,
    "fault_probability": 0.03,
    "warmup_cycles":     200,
    "window_size":       10,
    "seed":              42,
}

# ── Google Sheets tab names
SHEET_TABS = {
    "raw":      "01_RawSensorData",
    "features": "02_ProcessedFeatures",
    "preds":    "03_Predictions",
    "decisions":"04_DecisionSupport",
    "metrics":  "05_ModelMetrics",
}



# Helpers


def divider(title: str = ""):
    width = 70
    print("\n" + "─" * width)
    if title:
        print(f"  {title}")
        print("─" * width)


def build_metrics_df(reg_metrics: dict, cls_metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "model":  ["RUL Regressor",  "RUL Regressor",
                   "Maintenance Classifier", "Maintenance Classifier",
                   "Maintenance Classifier"],
        "metric": ["MSE",  "RMSE",  "Accuracy",  "Precision",  "Recall"],
        "value":  [reg_metrics["MSE"],  reg_metrics["RMSE"],
                   cls_metrics["Accuracy"], cls_metrics["Precision"],
                   cls_metrics["Recall"]],
    })


def save_to_sheets(writer, raw_df, feat_df, pred_df, decision_df, metrics_df):
    """Write all five DataFrames to named Google Sheets tabs."""
    from sheets_writer import SheetsWriter   # imported here so CSV fallback
                                             # works without gspread installed
    print("  Connecting to Google Sheets …")
    writer = SheetsWriter(CREDS_FILE, SHEET_ID)
    writer.write_many({
        SHEET_TABS["raw"]:      raw_df,
        SHEET_TABS["features"]: feat_df,
        SHEET_TABS["preds"]:    pred_df,
        SHEET_TABS["decisions"]:decision_df,
        SHEET_TABS["metrics"]:  metrics_df,
    })


def save_to_csv(raw_df, feat_df, pred_df, decision_df, metrics_df):
    """Fallback: write all DataFrames as CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = {
        f"{OUTPUT_DIR}/01_raw_sensor_data.csv":    raw_df,
        f"{OUTPUT_DIR}/02_processed_features.csv": feat_df,
        f"{OUTPUT_DIR}/03_predictions.csv":         pred_df,
        f"{OUTPUT_DIR}/04_decision_support.csv":    decision_df,
        f"{OUTPUT_DIR}/05_model_metrics.csv":       metrics_df,
    }
    for path, df in files.items():
        df.to_csv(path, index=False)
        size = os.path.getsize(path) / 1024
        print(f"  Saved: {os.path.basename(path)}  ({size:.1f} KB)")



# Main pipeline

def run_pipeline():
    t0 = time.time()
    print("=" * 70)
    print("  Real-Time IoT Data Pipeline — Execution Log")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output  : {'Google Sheets' if GOOGLE_SHEETS_ENABLED else 'CSV'}")
    print("=" * 70)

    # ── STAGE 1: DATA GENERATION
    divider("STAGE 1 — Sensor Data Simulation")
    sim = SensorSimulator(
        total_cycles      = CONFIG["total_cycles"],
        fault_probability = CONFIG["fault_probability"],
        seed              = CONFIG["seed"],
    )
    raw_df = sim.generate()
    print(f"  Generated {len(raw_df):,} sensor records")
    print(f"  Fault cycles : {raw_df['is_fault'].sum()} "
          f"({raw_df['is_fault'].mean()*100:.1f}%)")

    # ── STAGE 2: STREAM PROCESSING & FEATURE ENGINEERING ─────────────────────
    divider("STAGE 2 — Stream Processing & Feature Engineering")
    processor = StreamProcessor(window_size=CONFIG["window_size"])
    processor.fit_scaler(raw_df.head(CONFIG["warmup_cycles"]))
    print(f"  Scaler fitted on first {CONFIG['warmup_cycles']} records")

    feat_df = processor.process_dataframe(raw_df)
    print(f"  Processed records : {len(feat_df):,}")
    print(f"  Feature columns   : {len(feat_df.columns)}")

    # ── STAGE 3: MODEL TRAINING ───────────────────────────────────────────────
    divider("STAGE 3 — Model Training")

    rul_model   = RULRegressor()
    reg_metrics = rul_model.fit(feat_df)
    print(f"  [RUL Regressor]  MSE={reg_metrics['MSE']}  RMSE={reg_metrics['RMSE']}")

    cls_model   = MaintenanceClassifier(max_depth=5)
    cls_metrics = cls_model.fit(feat_df)
    print(f"  [Classifier]  Accuracy={cls_metrics['Accuracy']}  "
          f"Precision={cls_metrics['Precision']}  Recall={cls_metrics['Recall']}")

    # ── STAGE 4: PREDICTION ───────────────────────────────────────────────────
    divider("STAGE 4 — Prediction")
    rul_preds = rul_model.predict(feat_df)
    cls_preds = cls_model.predict(feat_df)
    print(f"  RUL  mean={rul_preds.mean():.1f}  "
          f"min={rul_preds.min():.0f}  max={rul_preds.max():.0f}")
    unique, counts = np.unique(cls_preds, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  Condition '{lbl}': {cnt} ({cnt/len(cls_preds)*100:.1f}%)")

    # ── STAGE 5: DECISION SUPPORT ─────────────────────────────────────────────
    divider("STAGE 5 — Decision Support")
    decisions   = evaluate_batch(feat_df, rul_preds, cls_preds)
    decision_df = pd.DataFrame(decisions)

    for level, count in decision_df["urgency"].value_counts().items():
        print(f"  {level:<16}: {count:4d} cycles")

    # ── BUILD OUTPUT DataFrames ───────────────────────────────────────────────
    pred_df = feat_df[["cycle", "timestamp", "rul",
                        "maintenance_flag", "replace_flag",
                        "health_index"]].copy()
    pred_df["rul_predicted"]        = rul_preds
    pred_df["condition_predicted"]  = cls_preds

    metrics_df = build_metrics_df(reg_metrics, cls_metrics)

    # ── STAGE 6: OUTPUT ───────────────────────────────────────────────────────
    divider("STAGE 6 — Writing Outputs")

    if GOOGLE_SHEETS_ENABLED:
        try:
            save_to_sheets(None, raw_df, feat_df, pred_df, decision_df, metrics_df)
        except Exception as e:
            print(f"  ⚠  Google Sheets write failed: {e}")
            print("  Falling back to CSV …")
            save_to_csv(raw_df, feat_df, pred_df, decision_df, metrics_df)
    else:
        save_to_csv(raw_df, feat_df, pred_df, decision_df, metrics_df)

    # ── DONE ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    divider()
    print(f"  Pipeline completed in {elapsed:.2f} seconds")
    print("=" * 70 + "\n")

    return {
        "raw_df":      raw_df,
        "feat_df":     feat_df,
        "pred_df":     pred_df,
        "decision_df": decision_df,
        "reg_metrics": reg_metrics,
        "cls_metrics": cls_metrics,
    }


if __name__ == "__main__":
    run_pipeline()
