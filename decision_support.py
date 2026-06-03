"""
decision_support.py
--------------------
Translates model predictions into actionable maintenance and spare-parts
recommendations aligned with the study's decision-support objective (Section 1.3).

This module does NOT make autonomous decisions; it provides structured
information to guide human operators.
"""

from dataclasses import dataclass, field
from datetime    import datetime


# ---------------------------------------------------------------------------
# Configuration thresholds (tunable without touching model code)
# ---------------------------------------------------------------------------

RUL_CRITICAL_THRESHOLD  = 50    # cycles — trigger replacement alert
RUL_WARNING_THRESHOLD   = 150   # cycles — order spare parts
HEALTH_CRITICAL         = 0.35  # health index below this = critical
HEALTH_WARNING          = 0.65  # health index below this = warning


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceRecommendation:
    cycle:           int
    timestamp:       datetime
    rul_predicted:   float
    health_index:    float
    condition:       str          # model classification
    urgency:         str          # "OK" | "Watch" | "Order Parts" | "Urgent"
    action:          str          # human-readable recommendation
    replace_flag:    int
    notes:           list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle":         self.cycle,
            "timestamp":     str(self.timestamp),
            "rul_predicted": self.rul_predicted,
            "health_index":  self.health_index,
            "condition":     self.condition,
            "urgency":       self.urgency,
            "action":        self.action,
            "replace_flag":  self.replace_flag,
            "notes":         "; ".join(self.notes),
        }


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Accepts per-record predictions and produces a MaintenanceRecommendation.
    """

    def evaluate(self,
                 cycle:         int,
                 timestamp:     datetime,
                 rul_predicted: float,
                 health_index:  float,
                 condition:     str,
                 replace_flag:  int) -> MaintenanceRecommendation:

        urgency, action, notes = self._derive_decision(
            rul_predicted, health_index, condition, replace_flag
        )

        return MaintenanceRecommendation(
            cycle=cycle,
            timestamp=timestamp,
            rul_predicted=rul_predicted,
            health_index=round(health_index, 4),
            condition=condition,
            urgency=urgency,
            action=action,
            replace_flag=replace_flag,
            notes=notes,
        )

    # ------------------------------------------------------------------

    def _derive_decision(self,
                         rul:          float,
                         health_index: float,
                         condition:    str,
                         replace_flag: int) -> tuple[str, str, list]:

        notes  = []
        urgency = "OK"
        action  = "Continue normal operation. No action required."

        # --- RUL-based logic ---
        if rul <= RUL_CRITICAL_THRESHOLD:
            urgency = "Urgent"
            action  = (f"Imminent failure risk — estimated {int(rul)} cycles remaining. "
                       "Schedule maintenance immediately and ensure spare parts are on hand.")
            notes.append(f"RUL={rul} is below critical threshold ({RUL_CRITICAL_THRESHOLD}).")

        elif rul <= RUL_WARNING_THRESHOLD:
            urgency = "Order Parts"
            action  = (f"Degradation detected — estimated {int(rul)} cycles remaining. "
                       "Initiate procurement of replacement components now to avoid downtime.")
            notes.append(f"RUL={rul} is below order threshold ({RUL_WARNING_THRESHOLD}).")

        # --- Health-index refinement ---
        if health_index < HEALTH_CRITICAL:
            if urgency not in ("Urgent",):
                urgency = "Urgent"
                action  = ("Component health is critically low. "
                           "Perform inspection and prepare for immediate replacement.")
            notes.append(f"Health index={health_index:.3f} below critical level ({HEALTH_CRITICAL}).")

        elif health_index < HEALTH_WARNING:
            if urgency == "OK":
                urgency = "Watch"
                action  = ("Mild degradation observed. "
                           "Increase monitoring frequency and plan a scheduled inspection.")
            notes.append(f"Health index={health_index:.3f} below warning level ({HEALTH_WARNING}).")

        # --- Condition label override ---
        if condition == "Maintenance Required" and urgency == "OK":
            urgency = "Urgent"
            action  = "Model indicates maintenance is required. Conduct an inspection promptly."
            notes.append("Classifier flagged: Maintenance Required.")

        elif condition == "Warning" and urgency == "OK":
            urgency = "Watch"
            action  = "Early warning signs detected. Monitor closely and prepare a maintenance plan."
            notes.append("Classifier flagged: Warning.")

        # --- Replace flag ---
        if replace_flag == 1:
            notes.append("Replace flag active: component replacement should be prioritised.")

        return urgency, action, notes


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def evaluate_batch(feat_df, rul_preds, cls_preds) -> list[dict]:
    """
    Run the DecisionEngine over every row in the processed feature DataFrame.

    Parameters
    ----------
    feat_df   : pd.DataFrame — output of StreamProcessor
    rul_preds : np.ndarray  — output of RULRegressor.predict(feat_df)
    cls_preds : np.ndarray  — output of MaintenanceClassifier.predict(feat_df)

    Returns
    -------
    List of dicts, one per row.
    """
    engine = DecisionEngine()
    results = []
    for i, row in feat_df.iterrows():
        idx = feat_df.index.get_loc(i)
        rec = engine.evaluate(
            cycle         = int(row["cycle"]),
            timestamp     = row["timestamp"],
            rul_predicted = float(rul_preds[idx]),
            health_index  = float(row["health_index"]),
            condition     = cls_preds[idx],
            replace_flag  = int(row["replace_flag"]),
        )
        results.append(rec.to_dict())
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    eng = DecisionEngine()

    test_cases = [
        dict(cycle=100, timestamp=datetime.now(), rul_predicted=300, health_index=0.80, condition="Normal",               replace_flag=0),
        dict(cycle=400, timestamp=datetime.now(), rul_predicted=130, health_index=0.55, condition="Warning",              replace_flag=0),
        dict(cycle=700, timestamp=datetime.now(), rul_predicted=45,  health_index=0.28, condition="Maintenance Required", replace_flag=1),
    ]

    for tc in test_cases:
        rec = eng.evaluate(**tc)
        print(f"Cycle {rec.cycle:4d} | Urgency: {rec.urgency:<12} | {rec.action}")
        for n in rec.notes:
            print(f"          NOTE: {n}")
        print()
