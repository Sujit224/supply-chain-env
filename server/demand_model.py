"""
Stochastic demand model for the Supply Chain RL Environment.

Demand is generated with three additive components:
    daily_demand[SKU] = base_demand[SKU]
                        × seasonal_factor(day, SKU)   # weekly + monthly cycles
                        × trend_factor(day, SKU)       # slow drift
                        + noise ~ Normal(0, σ[SKU])    # random daily variation
                        + spike if random() < 0.05     # 5 % demand spike

The forecast the agent receives is deliberately noisy — it sees the true
signal plus Gaussian noise — mirroring real-world forecasting tools.
"""

import math
import random
from typing import Dict, List

# ── SKU catalogue ─────────────────────────────────────────────────────────────

SKU_CONFIG: Dict[str, Dict] = {
    "SKU-A": {
        "description": "Fast-moving consumer good (FMCG)",
        "base_demand": 120,
        "sigma_fraction": 0.20,       # σ as fraction of base demand
        "seasonal_amplitude": 0.30,    # peak-to-trough swing
        "seasonal_period_days": 7,     # weekly seasonality
        "trend_per_day": 0.002,        # slight upward drift
        "spike_multiplier": 2.5,
        "lead_time_min": 3,
        "lead_time_max": 5,
        "unit_cost": 120,              # INR
        "holding_cost_per_unit_per_day": 0.5,
        "stockout_penalty_multiplier": 2.0,
        "perishable": False,
        "expiry_days": None,
    },
    "SKU-B": {
        "description": "Electronics component",
        "base_demand": 40,
        "sigma_fraction": 0.25,
        "seasonal_amplitude": 0.10,
        "seasonal_period_days": 30,
        "trend_per_day": 0.001,
        "spike_multiplier": 3.0,
        "lead_time_min": 14,
        "lead_time_max": 21,
        "unit_cost": 4500,
        "holding_cost_per_unit_per_day": 8.0,
        "stockout_penalty_multiplier": 2.0,
        "perishable": False,
        "expiry_days": None,
    },
    "SKU-C": {
        "description": "Raw material",
        "base_demand": 200,
        "sigma_fraction": 0.10,        # low variance — predictable
        "seasonal_amplitude": 0.05,
        "seasonal_period_days": 30,
        "trend_per_day": 0.0,
        "spike_multiplier": 1.5,
        "lead_time_min": 7,
        "lead_time_max": 10,
        "unit_cost": 800,
        "holding_cost_per_unit_per_day": 1.0,
        "stockout_penalty_multiplier": 2.0,
        "perishable": False,
        "expiry_days": None,
    },
    "SKU-D": {
        "description": "Perishable item",
        "base_demand": 95,
        "sigma_fraction": 0.22,
        "seasonal_amplitude": 0.20,
        "seasonal_period_days": 7,
        "trend_per_day": 0.001,
        "spike_multiplier": 2.0,
        "lead_time_min": 1,
        "lead_time_max": 2,
        "unit_cost": 60,
        "holding_cost_per_unit_per_day": 1.5,
        "stockout_penalty_multiplier": 2.0,
        "perishable": True,
        "expiry_days": 30,
    },
    "SKU-E": {
        "description": "Seasonal product",
        "base_demand": 50,
        "sigma_fraction": 0.30,
        "seasonal_amplitude": 0.60,    # large swing — low off-season, surge in-season
        "seasonal_period_days": 90,    # one full quarter = one season
        "trend_per_day": 0.003,
        "spike_multiplier": 2.0,
        "lead_time_min": 10,
        "lead_time_max": 15,
        "unit_cost": 2200,
        "holding_cost_per_unit_per_day": 4.0,
        "stockout_penalty_multiplier": 2.0,
        "perishable": False,
        "expiry_days": None,
    },
    "SKU-F": {
        "description": "New SKU — launched on Day 60 with zero demand history",
        "base_demand": 70,
        "sigma_fraction": 0.35,        # high uncertainty at launch
        "seasonal_amplitude": 0.15,
        "seasonal_period_days": 7,
        "trend_per_day": 0.005,        # ramp-up trend
        "spike_multiplier": 2.0,
        "lead_time_min": 5,
        "lead_time_max": 10,
        "unit_cost": 1500,
        "holding_cost_per_unit_per_day": 3.0,
        "stockout_penalty_multiplier": 2.0,
        "perishable": False,
        "expiry_days": None,
    },
}

FORECAST_NOISE_FRACTION = 0.15   # Gaussian noise added to forecasts seen by agent


class DemandModel:
    """
    Generates and forecasts stochastic daily demand for each active SKU.

    Args:
        active_skus: List of SKU keys active in this episode.
        seed: Random seed for reproducibility.
        surge_days: Tuple (start_day, end_day, multiplier) for a demand surge event,
                    or None if no surge in this episode.
    """

    def __init__(
        self,
        active_skus: List[str],
        seed: int = 42,
        surge_days: tuple = None,
    ) -> None:
        self._active_skus = active_skus
        self._rng = random.Random(seed)
        self._surge_days = surge_days          # (start_day, end_day, affected_skus, multiplier)
        self._demand_history: Dict[str, List[int]] = {sku: [] for sku in active_skus}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_demand(self, sku: str, day: int) -> int:
        """
        Sample true daily demand for a SKU on a given simulation day.

        Args:
            sku: SKU identifier.
            day: Simulation day (1-indexed).

        Returns:
            Non-negative integer demand in units.
        """
        cfg = SKU_CONFIG[sku]
        demand = self._compute_demand(cfg, day)

        # Apply surge multiplier if active for this SKU
        if self._surge_days is not None:
            start, end, surge_skus, mult = self._surge_days
            if start <= day <= end and sku in surge_skus:
                demand = int(demand * mult)

        demand = max(0, demand)
        self._demand_history[sku].append(demand)
        return demand

    def get_forecast(self, sku: str, current_day: int, horizon_days: int = 14) -> List[float]:
        """
        Return a noisy demand forecast for the next `horizon_days` days.

        The agent sees the true signal plus Gaussian noise, mirroring
        real forecasting tools (e.g. SAP APO).

        Args:
            sku: SKU identifier.
            current_day: Today's simulation day.
            horizon_days: Number of future days to forecast.

        Returns:
            List of float forecast values (may be non-integer due to noise).
        """
        cfg = SKU_CONFIG[sku]
        forecasts = []
        for offset in range(1, horizon_days + 1):
            future_day = current_day + offset
            true_demand = self._compute_demand(cfg, future_day)

            # Apply surge if known (surge dates are in openenv.yaml and visible to agent)
            if self._surge_days is not None:
                start, end, surge_skus, mult = self._surge_days
                if start <= future_day <= end and sku in surge_skus:
                    true_demand = true_demand * mult

            noise_sigma = true_demand * FORECAST_NOISE_FRACTION
            noisy = true_demand + self._rng.gauss(0, noise_sigma)
            forecasts.append(round(max(0.0, noisy), 1))

        return forecasts


    def get_history(self, sku: str, lookback_days: int = 30) -> List[int]:
        """
        Return actual recorded demand for the last N days.

        Args:
            sku: SKU identifier.
            lookback_days: Number of past days to return.

        Returns:
            List of integer demand values, oldest first.
        """
        history = self._demand_history.get(sku, [])
        return history[-lookback_days:]


    def average_daily_demand(self, sku: str) -> float:
        """Return the rolling average daily demand based on recorded history."""
        history = self._demand_history.get(sku, [])
        if not history:
            return float(SKU_CONFIG[sku]["base_demand"])
        return sum(history) / len(history)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_demand(self, cfg: Dict, day: int) -> int:
        """Compute the true demand value for a single (cfg, day) pair."""
        base = cfg["base_demand"]

        # Seasonal factor: sinusoidal cycle
        phase = 2 * math.pi * day / cfg["seasonal_period_days"]
        seasonal = 1.0 + cfg["seasonal_amplitude"] * math.sin(phase)

        # Trend factor: slow linear drift
        trend = 1.0 + cfg["trend_per_day"] * (day - 1)

        # Gaussian noise
        sigma = base * cfg["sigma_fraction"]
        noise = self._rng.gauss(0, sigma)

        # Demand spike: 5 % probability
        spike = 0
        if self._rng.random() < 0.05:
            spike = int(base * (cfg["spike_multiplier"] - 1))

        raw = base * seasonal * trend + noise + spike
        return max(0, round(raw))
