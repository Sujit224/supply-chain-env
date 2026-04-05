# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supplier model for the Supply Chain RL Environment.

Each SKU has three competing suppliers with different price / speed /
reliability trade-offs.  The agent must learn which supplier to use under
which conditions:

    Supplier Alpha  — lowest price, long lead time, highest reliability
    Supplier Beta   — medium price, medium lead time, medium reliability
    Supplier Gamma  — highest price (+25 %), shortest lead time, lower reliability
                      (emergency / expedite supplier)

Supplier reliability is the probability that an order arrives on time
and at full quantity.  When a reliability check fails, the order is
delayed by 3–7 days and/or arrives short by 10–40 %.
"""

import random
from typing import Dict, Optional, Tuple

from .demand_model import SKU_CONFIG

# ── Supplier catalogue ────────────────────────────────────────────────────────

SUPPLIER_CONFIG: Dict[str, Dict] = {
    "alpha": {
        "name": "Supplier Alpha",
        "price_multiplier": 1.00,      # baseline price
        "lead_time_min_offset": 8,     # added to SKU base lead time
        "lead_time_max_offset": 11,
        "reliability": 0.98,           # probability of on-time, full-quantity delivery
        "moq_multiplier": 2.5,         # minimum order quantity = SKU base_demand × multiplier
        "characteristic": "Reliable but slow — good for planned orders",
    },
    "beta": {
        "name": "Supplier Beta",
        "price_multiplier": 1.12,
        "lead_time_min_offset": 0,
        "lead_time_max_offset": 2,
        "reliability": 0.85,
        "moq_multiplier": 1.0,
        "characteristic": "Balanced — general purpose",
    },
    "gamma": {
        "name": "Supplier Gamma",
        "price_multiplier": 1.25,
        "lead_time_min_offset": -2,    # faster than SKU base lead time (min 1 day)
        "lead_time_max_offset": -1,
        "reliability": 0.80,
        "moq_multiplier": 0.25,        # low MOQ — good for small emergency orders
        "characteristic": "Expensive but fast — emergency replenishment",
    },
}

# Penalty parameters when a reliability check fails
DELAY_MIN_DAYS = 3
DELAY_MAX_DAYS = 7
SHORT_FILL_MIN = 0.60   # order may arrive at 60–90 % of requested quantity
SHORT_FILL_MAX = 0.90


class SupplierModel:
    """
    Models supplier behaviour including pricing, lead times, MOQs, and
    stochastic reliability failures.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed + 1000)   # offset to avoid correlation with demand seed
        self._degraded_suppliers: Dict[str, float] = {}   # supplier_id → new reliability

    # ── Public API ────────────────────────────────────────────────────────────

    def get_quotes(self, sku: str, quantity: int) -> Dict[str, Dict]:
        """
        Return price, lead time, and availability quotes from all three suppliers.

        Args:
            sku: SKU identifier.
            quantity: Requested order quantity (used to check MOQ).

        Returns:
            Dict keyed by supplier_id with price_per_unit, lead_time_days,
            reliability, moq, and available flag.
        """
        sku_cfg = SKU_CONFIG[sku]
        quotes = {}
        for sup_id, sup_cfg in SUPPLIER_CONFIG.items():
            moq = max(1, round(sku_cfg["base_demand"] * sup_cfg["moq_multiplier"]))
            lead_min = max(1, sku_cfg["lead_time_min"] + sup_cfg["lead_time_min_offset"])
            lead_max = max(lead_min, sku_cfg["lead_time_max"] + sup_cfg["lead_time_max_offset"])
            lead_time = self._rng.randint(lead_min, lead_max)
            price = round(sku_cfg["unit_cost"] * sup_cfg["price_multiplier"], 2)
            reliability = self._degraded_suppliers.get(sup_id, sup_cfg["reliability"])

            quotes[sup_id] = {
                "supplier_name": sup_cfg["name"],
                "price_per_unit": price,
                "lead_time_days": lead_time,
                "reliability": reliability,
                "moq": moq,
                "available": quantity >= moq,
                "characteristic": sup_cfg["characteristic"],
            }
        return quotes

    def resolve_order(
        self,
        sku: str,
        supplier_id: str,
        quantity: int,
        order_day: int,
    ) -> Tuple[int, int]:
        """
        Resolve an order: determine actual arrival day and quantity delivered.

        A reliability check is performed; on failure the order is delayed
        and/or arrives short.

        Args:
            sku: SKU identifier.
            supplier_id: One of 'alpha', 'beta', 'gamma'.
            quantity: Requested order quantity.
            order_day: Simulation day on which the order was placed.

        Returns:
            Tuple of (arrival_day, delivered_quantity).
        """
        sku_cfg = SKU_CONFIG[sku]
        sup_cfg = SUPPLIER_CONFIG[supplier_id]

        lead_min = max(1, sku_cfg["lead_time_min"] + sup_cfg["lead_time_min_offset"])
        lead_max = max(lead_min, sku_cfg["lead_time_max"] + sup_cfg["lead_time_max_offset"])
        base_lead = self._rng.randint(lead_min, lead_max)

        reliability = self._degraded_suppliers.get(supplier_id, sup_cfg["reliability"])
        if self._rng.random() <= reliability:
            # On-time, full quantity
            arrival_day = order_day + base_lead
            delivered_qty = quantity
        else:
            # Reliability failure: delayed and/or short
            delay = self._rng.randint(DELAY_MIN_DAYS, DELAY_MAX_DAYS)
            fill_fraction = self._rng.uniform(SHORT_FILL_MIN, SHORT_FILL_MAX)
            arrival_day = order_day + base_lead + delay
            delivered_qty = max(1, round(quantity * fill_fraction))

        return arrival_day, delivered_qty

    def apply_disruption(self, supplier_id: str, severity: float) -> None:
        """
        Degrade a supplier's effective reliability due to a disruption event.

        Args:
            supplier_id: Affected supplier.
            severity: Reliability reduction (0.0–1.0). A severity of 0.4 on a
                      0.85-reliability supplier reduces it to 0.51.
        """
        base = SUPPLIER_CONFIG[supplier_id]["reliability"]
        self._degraded_suppliers[supplier_id] = max(0.0, base - severity)

    def recover_disruption(self, supplier_id: str) -> None:
        """Restore a supplier to its baseline reliability after a disruption ends."""
        self._degraded_suppliers.pop(supplier_id, None)

    def get_supplier_reliability(self, supplier_id: str) -> float:
        """Return the current effective reliability for a supplier."""
        return self._degraded_suppliers.get(
            supplier_id, SUPPLIER_CONFIG[supplier_id]["reliability"]
        )

    def degrade_supplier_permanently(self, supplier_id: str, new_reliability: float) -> None:
        """
        Permanently degrade a supplier's reliability for the rest of the episode.
        Used in the hard task's reliability shock event.

        Args:
            supplier_id: Supplier to degrade.
            new_reliability: New reliability value (0.0–1.0).
        """
        self._degraded_suppliers[supplier_id] = max(0.0, new_reliability)

    @staticmethod
    def get_moq(sku: str, supplier_id: str) -> int:
        """Return the minimum order quantity for a given SKU / supplier pair."""
        sku_cfg = SKU_CONFIG[sku]
        sup_cfg = SUPPLIER_CONFIG[supplier_id]
        return max(1, round(sku_cfg["base_demand"] * sup_cfg["moq_multiplier"]))
