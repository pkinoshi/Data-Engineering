"""
sensor_simulator.py
--------------------
Simulates realistic IoT sensor readings for a single machine over time.
Readings include: temperature, vibration, pressure, rotational_speed.
Wear accumulates gradually; occasional fault spikes are injected.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class SensorSimulator:
    """
    Generates a continuous stream of sensor readings that mimic machine degradation.

    Parameters
    ----------
    total_cycles : int
        Total number of sensor readings to generate.
    fault_probability : float
        Probability of an anomalous spike at any given cycle.
    seed : int
        Random seed for reproducibility.
    """

    # --- Baseline operating ranges ---
    BASELINE = {
        "temperature":      {"mean": 70.0,  "std": 1.5},   # °C
        "vibration":        {"mean": 0.30,  "std": 0.02},  # g (acceleration)
        "pressure":         {"mean": 100.0, "std": 1.0},   # PSI
        "rotational_speed": {"mean": 1500.0,"std": 10.0},  # RPM
    }

    # --- How much each sensor degrades per cycle (linear drift) ---
    DEGRADATION_RATE = {
        "temperature":      0.005,
        "vibration":        0.0002,
        "pressure":        -0.003,
        "rotational_speed":-0.02,
    }

    # --- Fault spike magnitudes (added to baseline on fault cycles) ---
    FAULT_SPIKE = {
        "temperature":      15.0,
        "vibration":         0.30,
        "pressure":         20.0,
        "rotational_speed": -200.0,
    }

    def __init__(self, total_cycles: int = 1000,
                 fault_probability: float = 0.03,
                 seed: int = 42):
        self.total_cycles = total_cycles
        self.fault_probability = fault_probability
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full simulated dataset and return as a DataFrame."""
        records = []
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        for cycle in range(1, self.total_cycles + 1):
            wear_factor = cycle / self.total_cycles          # 0 → 1
            is_fault = self.rng.random() < self.fault_probability

            row = {"cycle": cycle,
                   "timestamp": start_time + timedelta(minutes=cycle * 5),
                   "is_fault": int(is_fault)}

            for sensor, base in self.BASELINE.items():
                drift = self.DEGRADATION_RATE[sensor] * cycle
                noise = self.rng.normal(0, base["std"])
                value = base["mean"] + drift + noise
                if is_fault:
                    value += self.FAULT_SPIKE[sensor]
                row[sensor] = round(value, 4)

            # Derived labels
            row["rul"]               = self._compute_rul(cycle)
            row["health_index"]      = self._compute_health_index(wear_factor)
            row["maintenance_flag"]  = self._maintenance_flag(row["health_index"])
            row["replace_flag"]      = int(row["rul"] < 50)

            records.append(row)

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rul(self, cycle: int) -> int:
        """Remaining Useful Life (cycles) with slight stochastic noise."""
        base_rul = max(0, self.total_cycles - cycle)
        noise = int(self.rng.integers(-5, 6))
        return max(0, base_rul + noise)

    def _compute_health_index(self, wear_factor: float) -> float:
        """
        Health index from 1.0 (perfect) to 0.0 (failed).
        Non-linear: degrades faster in the final third.
        """
        if wear_factor < 0.67:
            hi = 1.0 - wear_factor * 0.7
        else:
            hi = 1.0 - 0.67 * 0.7 - (wear_factor - 0.67) * 1.8
        return round(max(0.0, min(1.0, hi)), 4)

    def _maintenance_flag(self, health_index: float) -> str:
        """Map health index to a maintenance decision class."""
        if health_index >= 0.65:
            return "Normal"
        elif health_index >= 0.35:
            return "Warning"
        else:
            return "Maintenance Required"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = SensorSimulator(total_cycles=1000, fault_probability=0.04)
    df = sim.generate()
    print(df.to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Maintenance flag distribution:\n{df['maintenance_flag'].value_counts()}")
