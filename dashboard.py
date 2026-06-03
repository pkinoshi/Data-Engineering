"""
dashboard.py
------------
Interactive Streamlit Dashboard for the Real-Time IoT Data Pipeline.
Reads all data directly from Google Sheets (written by main_pipeline.py).

Setup
-----
1. Fill in CREDS_FILE and SHEET_ID below.
2. Run main_pipeline.py at least once to populate the sheets.
3. streamlit run dashboard.py

The sidebar has a Refresh button to re-pull from Sheets at any time.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Google Sheets configuration 
CREDS_FILE = "credentials.json"
SHEET_ID   = "1SO61B4KSnWAqKYUG3LocwspGEjZbR2_dF1gGue3S-K0"    

SHEET_TABS = {
    "raw":       "01_RawSensorData",
    "features":  "02_ProcessedFeatures",
    "preds":     "03_Predictions",
    "decisions": "04_DecisionSupport",
    "metrics":   "05_ModelMetrics",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Pipeline Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .urgency-ok     { color: #22c55e; font-weight: 700; }
    .urgency-watch  { color: #f59e0b; font-weight: 700; }
    .urgency-order  { color: #f97316; font-weight: 700; }
    .urgency-urgent { color: #ef4444; font-weight: 700; }
    .stMetric label { font-size: 0.78rem !important; }
    div[data-testid="stSidebarContent"] { background-color: #0f0f1a; }
    .source-badge {
        background: #1e3a5f; color: #60a5fa;
        border-radius: 6px; padding: 4px 10px;
        font-size: 0.75rem; font-weight: 600;
        display: inline-block; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "temp": "#f97316", "vib": "#a855f7", "pres": "#06b6d4", "rpm": "#22c55e",
    "rul":  "#3b82f6", "health": "#10b981", "fault": "#ef4444",
    "normal": "#22c55e", "warning": "#f59e0b", "maint": "#ef4444",
    "ok": "#22c55e", "watch": "#f59e0b", "order": "#f97316", "urgent": "#ef4444",
}
URGENCY_COLORS   = {"OK": C["ok"], "Watch": C["watch"],
                    "Order Parts": C["order"], "Urgent": C["urgent"]}
CONDITION_COLORS = {"Normal": C["normal"], "Warning": C["warning"],
                    "Maintenance Required": C["maint"]}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def apply_layout(fig, title="", height=340):
    fig.update_layout(**PLOTLY_LAYOUT, title=title,
                      title_font_size=14, height=height)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # cache for 5 min; Refresh clears it
def load_from_sheets(creds_file: str, sheet_id: str) -> dict:
    """
    Pull all five pipeline tabs from Google Sheets.
    Returns a dict of DataFrames or raises a descriptive error.
    """
    from sheets_writer import SheetsClient

    client = SheetsClient(creds_file, sheet_id)

    # Check which tabs actually exist before pulling
    available = client.available_tabs()
    missing   = [t for t in SHEET_TABS.values() if t not in available]
    if missing:
        raise FileNotFoundError(
            f"The following tabs were not found in your spreadsheet:\n"
            f"{missing}\n\n"
            f"Run  python main_pipeline.py  first to populate the sheet."
        )

    dfs = client.read_all(list(SHEET_TABS.values()))

    # ── type coercions so plots work correctly ────────────────────────
    raw  = dfs[SHEET_TABS["raw"]]
    feat = dfs[SHEET_TABS["features"]]
    pred = dfs[SHEET_TABS["preds"]]
    dec  = dfs[SHEET_TABS["decisions"]]
    met  = dfs[SHEET_TABS["metrics"]]

    for df, num_cols in [
        (raw,  ["temperature","vibration","pressure","rotational_speed",
                "cycle","rul","health_index","is_fault","replace_flag"]),
        (feat, ["cycle","health_index","rul","replace_flag",
                "temperature","vibration","pressure","rotational_speed",
                "temperature_roll_mean","vibration_roll_mean",
                "pressure_roll_mean","rotational_speed_roll_mean",
                "temperature_roll_std","vibration_roll_std",
                "pressure_roll_std","rotational_speed_roll_std"]),
        (pred, ["cycle","rul","rul_predicted","health_index","replace_flag"]),
        (dec,  ["cycle","rul_predicted","health_index","replace_flag"]),
        (met,  ["value"]),
    ]:
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return {"raw": raw, "feat": feat, "pred": pred,
            "dec": dec, "met": met}


def try_load(creds_file: str, sheet_id: str):
    """Wrapper that converts exceptions into Streamlit error messages."""
    try:
        return load_from_sheets(creds_file, sheet_id), None
    except FileNotFoundError as e:
        return None, ("pipeline_not_run", str(e))
    except Exception as e:
        return None, ("auth_error", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ IoT Pipeline Dashboard")
    st.markdown("---")

    #st.markdown("### 🔗 Google Sheets Source")
    creds_input = CREDS_FILE
    sheet_input = SHEET_ID

    #st.markdown("---")
    refresh_btn = st.button("🔄  Refresh from Sheets",
                            type="primary", use_container_width=True)
    if refresh_btn:
        load_from_sheets.clear()   # bust the cache

    st.markdown("---")
    st.markdown("### 📊 Cycle Range Filter")

    # Placeholder — will be updated once data loads
    cycle_range_placeholder = st.empty()

    st.markdown("---")
    st.caption("Data source: Google Sheets\nRefresh to pull latest pipeline run.")

# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚙️ Real-Time IoT Data Pipeline Dashboard")
#st.markdown('<span class="source-badge">📊 Data source: Google Sheets</span>',
#            unsafe_allow_html=True)
st.markdown("*Predictive maintenance · data pulled live from the spreadsheet*")
st.markdown("---")

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Connecting to Google Sheets …"):
    data, error = try_load(creds_input, sheet_input)

if error:
    kind, msg = error

    if kind == "pipeline_not_run":
        st.warning("⚠️  Pipeline output not found in Google Sheets.")
        st.info("Run the pipeline first, then come back here to view the dashboard:")
        st.code("python main_pipeline.py", language="bash")
        st.error(msg)

    else:
        st.error("❌  Could not connect to Google Sheets.")
        st.markdown("""
**Common causes:**
- `credentials.json` path is wrong or the file is missing
- The service account hasn't been granted access to the spreadsheet
- `SHEET_ID` is incorrect

**Fix:** Share the spreadsheet with your service account email
(found in `credentials.json` → `client_email`) as **Editor**.
        """)
        with st.expander("Full error details"):
            st.code(msg)

    st.stop()   # nothing else to render without data

# ── Data loaded — unpack ─────────────────────────────────────────────────────
raw_df, feat_df, pred_df, decision_df, metrics_df = (
    data["raw"], data["feat"], data["pred"], data["dec"], data["met"]
)

total_cycles = int(raw_df["cycle"].max())

# ── Cycle range filter (now we know the max) ─────────────────────────────────
with cycle_range_placeholder:
    cycle_range = st.slider("Cycle range", 1, total_cycles,
                            (1, total_cycles), step=10)

lo, hi = cycle_range
flt_raw  = raw_df[(raw_df["cycle"]  >= lo) & (raw_df["cycle"]  <= hi)].copy()
flt_feat = feat_df[(feat_df["cycle"] >= lo) & (feat_df["cycle"] <= hi)].copy()
flt_pred = pred_df[(pred_df["cycle"] >= lo) & (pred_df["cycle"] <= hi)].copy()
flt_dec  = decision_df[(decision_df["cycle"] >= lo) &
                       (decision_df["cycle"] <= hi)].copy()

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Cycles (filtered)",       f"{len(flt_feat):,}")
k2.metric("Fault Events",            f"{int(flt_raw['is_fault'].sum())}")
k3.metric("Urgent Alerts",           f"{int((flt_dec['urgency']=='Urgent').sum())}")
k4.metric("Avg RUL (predicted)",     f"{flt_pred['rul_predicted'].mean():.0f} cyc")
k5.metric("Avg Health Index",        f"{flt_feat['health_index'].mean():.3f}")
k6.metric("Replace Flag Rate",       f"{flt_pred['replace_flag'].mean()*100:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_sensor, tab_health, tab_pred, tab_decision, tab_model = st.tabs([
    "📡 Sensor Trends", "💚 Health & RUL", "🔮 Predictions",
    "🛠 Decision Support", "📈 Model Performance",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — SENSOR TRENDS
# ══════════════════════════════════════════════════════════════════
with tab_sensor:
    st.markdown("### Sensor Readings Over Time")
    sensor_select = st.multiselect(
        "Sensors", ["temperature","vibration","pressure","rotational_speed"],
        default=["temperature","vibration","pressure","rotational_speed"]
    )
    sensor_colors = {"temperature": C["temp"], "vibration": C["vib"],
                     "pressure": C["pres"], "rotational_speed": C["rpm"]}

    if sensor_select:
        fig = make_subplots(rows=len(sensor_select), cols=1, shared_xaxes=True,
                            vertical_spacing=0.04,
                            subplot_titles=[s.replace("_"," ").title()
                                            for s in sensor_select])
        faults = flt_raw[flt_raw["is_fault"] == 1]
        for i, sensor in enumerate(sensor_select, 1):
            fig.add_trace(go.Scatter(
                x=flt_raw["cycle"], y=flt_raw[sensor], mode="lines",
                name=sensor.replace("_"," ").title(),
                line=dict(color=sensor_colors[sensor], width=1.5),
            ), row=i, col=1)
            if not faults.empty:
                fig.add_trace(go.Scatter(
                    x=faults["cycle"], y=faults[sensor], mode="markers",
                    name="Fault" if i == 1 else None,
                    marker=dict(color=C["fault"], size=7, symbol="x"),
                    showlegend=(i == 1),
                ), row=i, col=1)
            fig.update_yaxes(gridcolor="#1e1e2e", row=i, col=1)

        fig.update_layout(**PLOTLY_LAYOUT,
                          height=150*len(sensor_select)+60,
                          title="Sensor Trends with Fault Events",
                          title_font_size=14)
        fig.update_xaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rolling Mean per Sensor")
    cols = st.columns(4)
    for col, (sensor, color) in zip(cols, [
        ("temperature", C["temp"]), ("vibration", C["vib"]),
        ("pressure", C["pres"]), ("rotational_speed", C["rpm"])
    ]):
        col_name = f"{sensor}_roll_mean"
        if col_name in flt_feat.columns:
            with col:
                fig = go.Figure(go.Scatter(
                    x=flt_feat["cycle"], y=flt_feat[col_name],
                    mode="lines", line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",

                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title=sensor.replace("_"," ").title(),
                                  title_font_size=13, height=200)
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — HEALTH & RUL
# ══════════════════════════════════════════════════════════════════
with tab_health:
    st.markdown("### Machine Health Index & Remaining Useful Life")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flt_feat["cycle"], y=flt_feat["health_index"],
            mode="lines", name="Health Index",
            line=dict(color=C["health"], width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.2)",
        ))
        fig.add_hline(y=0.65, line_dash="dash", line_color=C["warning"],
                      annotation_text="Warning", annotation_position="top right")
        fig.add_hline(y=0.35, line_dash="dash", line_color=C["urgent"],
                      annotation_text="Critical", annotation_position="top right")
        apply_layout(fig, "Health Index Degradation", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flt_pred["cycle"], y=flt_pred["rul"],
            mode="lines", name="Actual RUL",
            line=dict(color=C["rpm"], width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=flt_pred["cycle"], y=flt_pred["rul_predicted"],
            mode="lines", name="Predicted RUL",
            line=dict(color=C["rul"], width=2),
        ))
        fig.add_hline(y=150, line_dash="dash", line_color=C["order"],
                      annotation_text="Order Parts", annotation_position="top right")
        fig.add_hline(y=50, line_dash="dash", line_color=C["urgent"],
                      annotation_text="Critical", annotation_position="top right")
        apply_layout(fig, "Actual vs Predicted RUL", height=360)
        st.plotly_chart(fig, use_container_width=True)

    phase_counts = flt_feat["maintenance_flag"].value_counts() \
        if "maintenance_flag" in flt_feat.columns \
        else flt_pred["condition_predicted"].value_counts()
    fig = go.Figure(go.Pie(
        labels=phase_counts.index, values=phase_counts.values,
        marker_colors=[CONDITION_COLORS.get(l, "#888") for l in phase_counts.index],
        hole=0.5, textinfo="label+percent",
    ))
    apply_layout(fig, "Operating Condition Distribution", height=320)
    fig.update_layout(legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### Prediction Output")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flt_pred["rul"], y=flt_pred["rul_predicted"],
            mode="markers",
            marker=dict(color=C["rul"], size=4, opacity=0.5),
            name="Predictions",
        ))
        max_val = max(flt_pred["rul"].max(), flt_pred["rul_predicted"].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            name="Perfect fit", line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        ))
        apply_layout(fig, "Actual vs Predicted RUL (Scatter)", height=360)
        fig.update_xaxes(title="Actual RUL")
        fig.update_yaxes(title="Predicted RUL")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cond_map = {"Normal": 0, "Warning": 1, "Maintenance Required": 2}
        flt_pred_c = flt_pred.copy()
        flt_pred_c["cond_num"] = flt_pred_c["condition_predicted"].map(cond_map)
        fig = go.Figure()
        for label, num in cond_map.items():
            sub = flt_pred_c[flt_pred_c["cond_num"] == num]
            fig.add_trace(go.Scatter(
                x=sub["cycle"], y=sub["cond_num"], mode="markers",
                marker=dict(color=CONDITION_COLORS[label], size=4, opacity=0.7),
                name=label,
            ))
        fig.update_yaxes(tickvals=[0,1,2],
                         ticktext=["Normal","Warning","Maint. Required"])
        apply_layout(fig, "Predicted Condition Over Time", height=360)
        st.plotly_chart(fig, use_container_width=True)

    flt_pred_r = flt_pred.copy()
    flt_pred_r["residual"] = flt_pred_r["rul"] - flt_pred_r["rul_predicted"]
    fig = go.Figure(go.Bar(
        x=flt_pred_r["cycle"], y=flt_pred_r["residual"],
        marker_color=[C["rul"] if r >= 0 else C["urgent"]
                      for r in flt_pred_r["residual"]],
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    apply_layout(fig, "RUL Residuals (Actual − Predicted)", height=260)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prediction Data")
    st.dataframe(flt_pred.head(200).style.format({
        "rul": "{:.0f}", "rul_predicted": "{:.1f}", "health_index": "{:.4f}",
    }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — DECISION SUPPORT
# ══════════════════════════════════════════════════════════════════
with tab_decision:
    st.markdown("### Maintenance Decision Support")
    c1, c2 = st.columns([2, 1])

    with c1:
        urg_map = {"OK": 0, "Watch": 1, "Order Parts": 2, "Urgent": 3}
        flt_dec_c = flt_dec.copy()
        flt_dec_c["urg_num"] = flt_dec_c["urgency"].map(urg_map)
        fig = go.Figure()
        for label, num in urg_map.items():
            sub = flt_dec_c[flt_dec_c["urg_num"] == num]
            fig.add_trace(go.Scatter(
                x=sub["cycle"], y=sub["urg_num"], mode="markers",
                marker=dict(color=URGENCY_COLORS[label], size=5, opacity=0.8),
                name=label,
            ))
        fig.update_yaxes(tickvals=[0,1,2,3],
                         ticktext=["OK","Watch","Order Parts","Urgent"])
        apply_layout(fig, "Urgency Level Timeline", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        urg_counts = flt_dec["urgency"].value_counts()
        fig = go.Figure(go.Pie(
            labels=urg_counts.index, values=urg_counts.values,
            marker_colors=[URGENCY_COLORS.get(l,"#888") for l in urg_counts.index],
            hole=0.55, textinfo="label+percent",
        ))
        apply_layout(fig, "Urgency Distribution", height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Rolling condition proportions
    window = 50
    tmp = flt_dec.copy().set_index("cycle")
    for cond in ["Normal", "Warning", "Maintenance Required"]:
        key = f"_roll_{cond.replace(' ','_')}"
        tmp[key] = (tmp["condition"] == cond).astype(float) \
                    .rolling(window, min_periods=1).mean()
    fig = go.Figure()
    for cond, color in CONDITION_COLORS.items():
        key = f"_roll_{cond.replace(' ','_')}"
        fig.add_trace(go.Scatter(
            x=tmp.index, y=tmp[key], mode="lines", name=cond,
            line=dict(color=color, width=2),
            stackgroup="one", fillcolor="rgba(34,197,94,0.4)",
        ))
    apply_layout(fig, f"Rolling {window}-Cycle Condition Proportion", height=280)
    fig.update_yaxes(title="Proportion", tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚠️ Urgent Alerts")
    urgent = flt_dec[flt_dec["urgency"] == "Urgent"][
        ["cycle","rul_predicted","health_index","condition","urgency","action","notes"]
    ].head(100)
    if urgent.empty:
        st.info("No urgent alerts in the selected cycle range.")
    else:
        st.dataframe(urgent.style.format({
            "rul_predicted": "{:.0f}", "health_index": "{:.4f}",
        }), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### Model Evaluation Metrics")

    # Pull values from the metrics sheet
    def get_metric(name: str) -> float:
        row = metrics_df[metrics_df["metric"] == name]
        return float(row["value"].values[0]) if not row.empty else float("nan")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📐 RUL Regressor (Linear Regression)")
        m1, m2 = st.columns(2)
        m1.metric("MSE",  f"{get_metric('MSE'):.4f}")
        m2.metric("RMSE", f"{get_metric('RMSE'):.4f}")

        flt_pred_r2 = flt_pred.copy()
        flt_pred_r2["residual"] = flt_pred_r2["rul"] - flt_pred_r2["rul_predicted"]
        fig = go.Figure(go.Histogram(
            x=flt_pred_r2["residual"], nbinsx=40,
            marker_color=C["rul"], opacity=0.8,
        ))
        apply_layout(fig, "RUL Residual Distribution", height=260)
        fig.update_xaxes(title="Residual (cycles)")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🌳 Maintenance Classifier (Decision Tree, depth=5)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  f"{get_metric('Accuracy'):.4f}")
        m2.metric("Precision", f"{get_metric('Precision'):.4f}")
        m3.metric("Recall",    f"{get_metric('Recall'):.4f}")

        # Confusion matrix derived from prediction data
        from sklearn.metrics import confusion_matrix
        labels = ["Normal", "Warning", "Maintenance Required"]
        valid  = flt_pred.dropna(subset=["maintenance_flag","condition_predicted"])
        if not valid.empty:
            cm = confusion_matrix(valid["maintenance_flag"],
                                  valid["condition_predicted"], labels=labels)
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Normal","Warning","Maint. Req."],
                y=["Normal","Warning","Maint. Req."],
                colorscale="Blues", text=cm, texttemplate="%{text}",
            ))
            apply_layout(fig, "Confusion Matrix", height=320)
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough labelled data in range for confusion matrix.")

    st.markdown("---")
    st.markdown("### 📋 Raw Metrics from Google Sheets")
    st.dataframe(metrics_df, use_container_width=True)

    # Last-updated info
    st.markdown("---")
    st.caption(f"Data pulled from Google Sheets · Spreadsheet ID: `{sheet_input}` · "
               "Use the **Refresh** button in the sidebar to reload.")
