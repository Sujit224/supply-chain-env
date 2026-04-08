# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Environment Implementation.

Simulates a multi-product, multi-supplier supply chain over a configurable
episode horizon (30 / 60 / 90 days depending on task).  The agent interacts
entirely via MCP tool calls, matching how a real supply chain manager would
use enterprise software such as SAP or Oracle SCM.

Three graded tasks of escalating difficulty:
    easy   — Single-SKU (SKU-C), 30 days, 1 supplier, no disruptions.
    medium — All 5 SKUs, 60 days, 3 suppliers, 1 surprise disruption.
    hard   — All 6 SKUs, 90 days, 3 suppliers, 2-3 disruptions, demand surge,
             cash constraint, reliability shock, new-SKU launch on Day 60.
"""

import math
import random
from statistics import mean
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupplyChainAction, SupplyChainObservation
    from .demand_model import SKU_CONFIG, DemandModel
    from .supplier_models import SUPPLIER_CONFIG, SupplierModel
except ImportError:
    from models import SupplyChainAction, SupplyChainObservation
    from server.demand_model import SKU_CONFIG, DemandModel
    from server.supplier_models import SUPPLIER_CONFIG, SupplierModel

# ── Task configuration ────────────────────────────────────────────────────────

TASK_CONFIG: Dict[str, Dict] = {
    "easy": {
        "duration_days": 30,
        "active_skus": ["SKU-C"],
        "active_suppliers": ["alpha"],
        "warehouse_capacity": 2000,
        "monthly_cash_limit": None,
        "disruptions": [],
        "surge_days": None,
        "new_sku_launch_day": None,
        "reliability_shock": None,
    },
    "medium": {
        "duration_days": 60,
        "active_skus": ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        "active_suppliers": ["alpha", "beta", "gamma"],
        "warehouse_capacity": 10000,
        "monthly_cash_limit": 1_500_000,
        "disruptions": 1,
        "surge_days": None,
        "new_sku_launch_day": None,
        "reliability_shock": None,
    },
    "hard": {
        "duration_days": 90,
        "active_skus": ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        "active_suppliers": ["alpha", "beta", "gamma"],
        "warehouse_capacity": 10000,
        "monthly_cash_limit": 1_200_000,
        "disruptions": 3,
        "surge_days": (40, 44, ["SKU-A", "SKU-D"], 3.0),
        "new_sku_launch_day": 60,
        "reliability_shock": ("beta", 0.45, 45),
    },
}

STARTING_INVENTORY_DAYS = 10

STARTING_CASH: Dict[str, float] = {
    "easy": 5_000_000.0,
    "medium": 3_000_000.0,
    "hard": 2_500_000.0,
}

REWARD_FILL_PER_UNIT_WEIGHT: Dict[str, float] = {
    "SKU-A": 0.25, "SKU-B": 0.30, "SKU-C": 0.15,
    "SKU-D": 0.20, "SKU-E": 0.10, "SKU-F": 0.08,
}
REWARD_PROACTIVE_ORDER_BONUS = 0.5
REWARD_PANIC_ORDER_PENALTY = -0.3
REWARD_PERISHABLE_EXPIRY_MULTIPLIER = 2.0
REWARD_WAREHOUSE_OVERFLOW_PENALTY = -5.0
REWARD_CASH_BREACH_PENALTY = -10.0
REWARD_DISRUPTION_PREEMPTED_BONUS = 3.0
REWARD_SUPPLIER_RELIABILITY_BONUS = 0.2
TERMINAL_REWARD_SCALE = 50.0


class SupplyChainEnvironment(Environment):
    """
    Full supply chain simulator.

    The agent interacts exclusively via MCP tool calls (SupplyChainAction).
    State is updated after every action and every daily tick.

    Example:
        >>> env = SupplyChainEnvironment()
        >>> obs = env.reset(task_id="easy", seed=0)
        >>> obs = env.step(SupplyChainAction(tool_name="observe_inventory"))
        >>> obs = env.step(SupplyChainAction(
        ...     tool_name="place_order",
        ...     sku="SKU-C",
        ...     supplier_id="alpha",
        ...     quantity=500,
        ...     reason="Replenishing below reorder point",
        ... ))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialise with a blank state; call reset() before stepping."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "easy"
        self._seed: int = 42
        self._day: int = 1
        self._done: bool = False

        self._demand_model: Optional[DemandModel] = None
        self._supplier_model: Optional[SupplierModel] = None

        self._inventory: Dict[str, int] = {}
        self._pending_orders: List[Dict[str, Any]] = []
        self._order_history: List[Dict[str, Any]] = []
        self._reorder_points: Dict[str, int] = {}
        self._safety_stock: Dict[str, int] = {}

        self._cash_balance: float = 0.0
        self._month_spend: float = 0.0
        self._month_start_day: int = 1

        self._total_carrying_cost: float = 0.0
        self._total_stockout_cost: float = 0.0
        self._total_procurement_cost: float = 0.0

        self._daily_demand: Dict[str, List[int]] = {}
        self._daily_filled: Dict[str, List[int]] = {}
        self._daily_inventory: Dict[str, List[int]] = {}
        self._daily_warehouse_util: List[float] = []
        self._daily_cash_balance: List[float] = []
        self._stockout_history: Dict[str, int] = {}
        self._total_expired: Dict[str, int] = {}

        self._active_disruptions: List[Dict[str, Any]] = []
        self._episode_disruptions: List[Dict[str, Any]] = []
        self._disruption_schedule: List[Dict[str, Any]] = []

        self._cfg: Dict[str, Any] = {}
        self._active_skus: List[str] = []
        self._order_counter: int = 0

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: int = 42) -> SupplyChainObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'easy', 'medium', 'hard'.
            seed: Random seed for reproducibility.

        Returns:
            Initial SupplyChainObservation.
        """
        self._task_id = task_id
        self._seed = seed
        self._cfg = TASK_CONFIG[task_id]
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._day = 1
        self._done = False
        self._order_counter = 0

        self._active_skus = list(self._cfg["active_skus"])

        self._demand_model = DemandModel(
            active_skus=self._active_skus,
            seed=seed,
            surge_days=self._cfg.get("surge_days"),
        )
        self._supplier_model = SupplierModel(seed=seed)

        self._inventory = {
            sku: SKU_CONFIG[sku]["base_demand"] * STARTING_INVENTORY_DAYS
            for sku in self._active_skus
        }
        self._reorder_points = {
            sku: SKU_CONFIG[sku]["base_demand"] * SKU_CONFIG[sku]["lead_time_max"]
            for sku in self._active_skus
        }
        self._safety_stock = {
            sku: round(SKU_CONFIG[sku]["base_demand"] * 0.8)
            for sku in self._active_skus
        }
        self._pending_orders = []
        self._order_history = []

        self._cash_balance = STARTING_CASH[task_id]
        self._month_spend = 0.0
        self._month_start_day = 1

        self._total_carrying_cost = 0.0
        self._total_stockout_cost = 0.0
        self._total_procurement_cost = 0.0
        self._daily_demand = {sku: [] for sku in self._active_skus}
        self._daily_filled = {sku: [] for sku in self._active_skus}
        self._daily_inventory = {sku: [] for sku in self._active_skus}
        self._daily_warehouse_util = []
        self._daily_cash_balance = []
        self._stockout_history = {sku: 0 for sku in self._active_skus}
        self._total_expired = {sku: 0 for sku in self._active_skus}

        self._disruption_schedule = self._build_disruption_schedule()
        self._active_disruptions = []
        self._episode_disruptions = []

        self._tick_day()

        return self._build_observation(
            tool_name="reset",
            tool_result={"message": f"Supply Chain environment ready. Task: {task_id}. Day 1 of {self._cfg['duration_days']}."},
        )

    def _squash_reward(self, raw_reward: float) -> float:
        """
        Squashes a raw reward to be strictly between 0 and 1.
        Uses a sigmoid function and clamps to avoid exact 0.0 or 1.0 due to float precision.
        """
        try:
            val = 1.0 / (1.0 + math.exp(-raw_reward))
        except OverflowError:
            # Handle math.exp overflow if raw_reward is a large negative number
            val = 0.0 if raw_reward < 0 else 1.0
            
        # Clamp bounds strictly inside (0, 1)
        return max(0.0001, min(0.9999, val))

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """
        Execute one MCP tool call and advance the simulation by one step.

        Args:
            action: SupplyChainAction specifying the tool to call and its parameters.

        Returns:
            SupplyChainObservation with the tool result, updated state, and reward.
        """
        if self._done:
            return self._build_observation(
                tool_name=action.tool_name,
                tool_result={"error": "Episode is done. Call reset() to start a new episode."},
                reward=self._squash_reward(0.0), # Ensures it adheres to bounds even if called after done
            )

        self._state.step_count += 1

        # Normalize tool names defensively so read-only tools never
        # accidentally consume simulation time due to casing/whitespace.
        normalized_tool_name = (action.tool_name or "").strip().lower()
        action.tool_name = normalized_tool_name

        tool_result, step_reward = self._dispatch_tool(action)

        OBSERVATION_TOOLS = {
            "observe_inventory", "get_demand_forecast", "get_demand_history",
            "get_supplier_quotes", "check_warehouse_capacity", "observe_disruptions"
        }

        is_observation_tool = normalized_tool_name in OBSERVATION_TOOLS
        if not is_observation_tool:
            self._day += 1
            if self._day <= self._cfg["duration_days"]:
                daily_reward = self._tick_day()
                step_reward += daily_reward

        if self._day >= self._cfg["duration_days"]:
            self._done = True

        grade_score = None
        if self._done:
            grade_score = self._grade()
            step_reward += grade_score * TERMINAL_REWARD_SCALE

        # Final squashing before returning to guarantee 0 < reward < 1 at every step
        squashed_step_reward = self._squash_reward(step_reward)

        return self._build_observation(
            tool_name=action.tool_name,
            tool_result=tool_result,
            reward=squashed_step_reward,
            grade_score=grade_score,
        )

    @property
    def state(self) -> State:
        """Return the current environment state."""
        return self._state

    # ── Daily simulation tick ─────────────────────────────────────────────────

    def _tick_day(self) -> float:
        """
        Advance the simulation by one day and calculate dense rewards.

        Order of operations:
            1. Receive any orders arriving today.
            2. Generate and fulfil demand.
            3. Expire perishables past their shelf life.
            4. Charge holding costs.
            5. Roll the monthly budget if needed.
            6. Trigger / resolve disruptions per schedule.
            7. Handle hard-task events (new SKU launch, reliability shock).
            8. Record daily snapshots.
        """
        daily_reward = 0.0
        cfg = self._cfg

        self._receive_orders()
        daily_reward += self._fulfil_demand()
        daily_reward += self._expire_perishables()
        self._charge_holding_costs()

        if self._day - self._month_start_day >= 30:
            self._month_spend = 0.0
            self._month_start_day = self._day

        self._process_disruption_schedule()

        if cfg.get("new_sku_launch_day") and self._day == cfg["new_sku_launch_day"]:
            self._launch_new_sku("SKU-F")
        shock = cfg.get("reliability_shock")
        if shock and self._day == shock[2]:
            self._supplier_model.degrade_supplier_permanently(shock[0], shock[1])

        self._record_snapshots()
        
        return daily_reward

    def _receive_orders(self) -> None:
        """Move orders whose ETA matches today into on-hand inventory."""
        still_pending = []
        for order in self._pending_orders:
            if order["eta_day"] <= self._day:
                sku = order["sku"]
                qty = order["delivered_quantity"]
                if sku in self._inventory:
                    space = self._cfg["warehouse_capacity"] - self._total_inventory()
                    received = min(qty, space)
                    self._inventory[sku] += received
            else:
                still_pending.append(order)
        self._pending_orders = still_pending

    def _fulfil_demand(self) -> float:
        """Sample demand for each active SKU and fill as much as possible."""
        daily_fill_reward = 0.0
        for sku in self._active_skus:
            demand = self._demand_model.get_demand(sku, self._day)
            self._daily_demand[sku].append(demand)

            available = self._inventory.get(sku, 0)
            filled = min(demand, available)
            unfilled = demand - filled

            self._inventory[sku] = available - filled
            self._daily_filled[sku].append(filled)
            
            daily_fill_reward += filled * REWARD_FILL_PER_UNIT_WEIGHT.get(sku, 0.0)

            if unfilled > 0:
                self._stockout_history[sku] = self._stockout_history.get(sku, 0) + 1
                penalty = unfilled * SKU_CONFIG[sku]["unit_cost"] * SKU_CONFIG[sku]["stockout_penalty_multiplier"]
                self._total_stockout_cost += penalty
        
        return daily_fill_reward

    def _expire_perishables(self) -> float:
        """Discard inventory that has exceeded its shelf life."""
        expiry_penalty = 0.0
        for sku in self._active_skus:
            cfg = SKU_CONFIG[sku]
            if not cfg["perishable"]:
                continue
            expiry = cfg["expiry_days"]
            avg_age_days = expiry / 2
            if avg_age_days >= expiry:
                expired = self._inventory.get(sku, 0)
                self._inventory[sku] = 0
                self._total_expired[sku] = self._total_expired.get(sku, 0) + expired
                expiry_penalty -= expired * cfg["unit_cost"] * REWARD_PERISHABLE_EXPIRY_MULTIPLIER
                
            days_in_episode = self._day
            if days_in_episode > expiry:
                excess_fraction = min(0.05, (days_in_episode - expiry) / (expiry * 10))
                to_expire = round(self._inventory.get(sku, 0) * excess_fraction)
                self._inventory[sku] = max(0, self._inventory.get(sku, 0) - to_expire)
                self._total_expired[sku] = self._total_expired.get(sku, 0) + to_expire
                expiry_penalty -= to_expire * cfg["unit_cost"] * REWARD_PERISHABLE_EXPIRY_MULTIPLIER
                
        return expiry_penalty

    def _charge_holding_costs(self) -> None:
        """Charge daily holding (carrying) costs for all on-hand inventory."""
        daily_cost = 0.0
        for sku in self._active_skus:
            units = self._inventory.get(sku, 0)
            cost = units * SKU_CONFIG[sku]["holding_cost_per_unit_per_day"]
            daily_cost += cost
        self._total_carrying_cost += daily_cost

    def _record_snapshots(self) -> None:
        """Record per-day snapshots used by the graders and reward function."""
        for sku in self._active_skus:
            self._daily_inventory[sku].append(self._inventory.get(sku, 0))

        util = self._total_inventory() / max(1, self._cfg["warehouse_capacity"])
        self._daily_warehouse_util.append(util)
        self._daily_cash_balance.append(self._cash_balance)

    # ── MCP tool dispatcher ───────────────────────────────────────────────────

    def _dispatch_tool(self, action: SupplyChainAction):
        """Route the action to the correct tool handler. Returns (result_dict, reward)."""
        tool = action.tool_name
        reward = 0.0

        if tool == "observe_inventory":
            result = self._tool_observe_inventory(action.sku)
        elif tool == "get_demand_forecast":
            result = self._tool_get_demand_forecast(action.sku, action.horizon_days or 14)
        elif tool == "get_demand_history":
            result = self._tool_get_demand_history(action.sku, action.lookback_days or 30)
        elif tool == "get_supplier_quotes":
            result = self._tool_get_supplier_quotes(action.sku, action.quantity or 100)
        elif tool == "place_order":
            result, reward = self._tool_place_order(action)
        elif tool == "cancel_order":
            result = self._tool_cancel_order(action.order_id, action.reason)
        elif tool == "adjust_reorder_point":
            result = self._tool_adjust_reorder_point(action.sku, action.new_threshold, action.reason)
        elif tool == "adjust_safety_stock":
            result = self._tool_adjust_safety_stock(action.sku, action.new_level, action.reason)
        elif tool == "check_warehouse_capacity":
            result = self._tool_check_warehouse_capacity()
        elif tool == "expedite_order":
            result, reward = self._tool_expedite_order(action.order_id, action.reason)
        elif tool == "transfer_between_skus":
            result = self._tool_transfer_between_skus(action.from_sku, action.to_sku, action.quantity)
        elif tool == "observe_disruptions":
            result = self._tool_observe_disruptions()
        else:
            result = {"error": f"Unknown tool: {tool}. Check openenv.yaml for the valid tool list."}

        return result, reward

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_observe_inventory(self, sku: Optional[str]) -> Dict:
        if sku:
            if sku not in self._active_skus:
                return {"error": f"{sku} is not active in this task."}
            in_transit = [o for o in self._pending_orders if o["sku"] == sku]
            return {
                "sku": sku,
                "on_hand": self._inventory.get(sku, 0),
                "in_transit": in_transit,
                "reorder_point": self._reorder_points.get(sku),
                "safety_stock": self._safety_stock.get(sku),
                "stockout_days": self._stockout_history.get(sku, 0),
                "fill_rate": self._compute_fill_rate(sku),
            }
        return {
            "inventory": dict(self._inventory),
            "reorder_points": dict(self._reorder_points),
            "safety_stock": dict(self._safety_stock),
            "pending_orders": self._pending_orders,
            "stockout_history": dict(self._stockout_history),
            "fill_rates": {sku: self._compute_fill_rate(sku) for sku in self._active_skus},
        }

    def _tool_get_demand_forecast(self, sku: Optional[str], horizon_days: int) -> Dict:
        if sku:
            if sku not in self._active_skus:
                return {"error": f"{sku} is not active in this task."}
            forecast = self._demand_model.get_forecast(sku, self._day, horizon_days)
            avg = self._demand_model.average_daily_demand(sku)
            return {
                "sku": sku,
                "horizon_days": horizon_days,
                "forecast": forecast,
                "average_daily_demand": round(avg, 1),
                "note": "Forecast includes noise — actual demand will vary.",
            }
        results = {}
        for s in self._active_skus:
            results[s] = self._demand_model.get_forecast(s, self._day, horizon_days)
        return {"horizon_days": horizon_days, "forecasts": results}

    def _tool_get_demand_history(self, sku: Optional[str], lookback_days: int) -> Dict:
        if sku:
            if sku not in self._active_skus:
                return {"error": f"{sku} is not active in this task."}
            history = self._demand_model.get_history(sku, lookback_days)
            avg = round(sum(history) / len(history), 1) if history else 0
            return {"sku": sku, "lookback_days": lookback_days, "history": history, "average": avg}
        return {
            s: self._demand_model.get_history(s, lookback_days)
            for s in self._active_skus
        }

    def _tool_get_supplier_quotes(self, sku: Optional[str], quantity: int) -> Dict:
        if sku:
            if sku not in self._active_skus:
                return {"error": f"{sku} is not active in this task."}
            quotes = self._supplier_model.get_quotes(sku, quantity)
            active_quotes = {
                sup_id: q for sup_id, q in quotes.items()
                if sup_id in self._cfg["active_suppliers"]
            }
            return {"sku": sku, "requested_quantity": quantity, "quotes": active_quotes}
        results = {}
        for s in self._active_skus:
            quotes = self._supplier_model.get_quotes(s, quantity)
            results[s] = {
                sup_id: q for sup_id, q in quotes.items()
                if sup_id in self._cfg["active_suppliers"]
            }
        return {"requested_quantity": quantity, "quotes_by_sku": results}

    def _tool_place_order(self, action: SupplyChainAction):
        sku = action.sku
        supplier_id = action.supplier_id
        quantity = action.quantity

        if not sku or sku not in self._active_skus:
            return {"error": f"Invalid or inactive SKU: {sku}"}, 0.0
        if not supplier_id or supplier_id not in self._cfg["active_suppliers"]:
            return {"error": f"Invalid or inactive supplier: {supplier_id}"}, 0.0
        if not quantity or quantity <= 0:
            return {"error": "quantity must be a positive integer."}, 0.0

        # Check warehouse capacity
        available_space = self._cfg["warehouse_capacity"] - self._total_inventory()
        if quantity > available_space:
            return {
                "error": f"Order rejected: warehouse overflow. Available space: {available_space} units.",
                "penalty": REWARD_WAREHOUSE_OVERFLOW_PENALTY,
            }, REWARD_WAREHOUSE_OVERFLOW_PENALTY

        # Get quotes and check price
        quotes = self._supplier_model.get_quotes(sku, quantity)
        if supplier_id not in quotes:
            return {"error": f"Supplier {supplier_id} not available."}, 0.0
        price_per_unit = quotes[supplier_id]["price_per_unit"]
        total_cost = price_per_unit * quantity

        # Check monthly cash limit
        monthly_limit = self._cfg.get("monthly_cash_limit")
        if monthly_limit and (self._month_spend + total_cost) > monthly_limit:
            return {
                "error": f"Order rejected: monthly cash limit exceeded. Remaining this month: {monthly_limit - self._month_spend:.0f} INR.",
                "penalty": REWARD_CASH_BREACH_PENALTY,
            }, REWARD_CASH_BREACH_PENALTY

        # Resolve order with supplier reliability
        arrival_day, delivered_qty = self._supplier_model.resolve_order(
            sku, supplier_id, quantity, self._day
        )

        order_id = f"PO-{self._order_counter:04d}"
        self._order_counter += 1

        self._pending_orders.append({
            "order_id": order_id,
            "sku": sku,
            "supplier_id": supplier_id,
            "requested_quantity": quantity,
            "delivered_quantity": delivered_qty,
            "eta_day": arrival_day,
            "order_day": self._day,
            "total_cost": total_cost,
        })
        self._order_history.append(self._pending_orders[-1])

        self._cash_balance -= total_cost
        self._month_spend += total_cost
        self._total_procurement_cost += total_cost

        # ── Reward shaping ────────────────────────────────────────────────────
        disruption_active = len(self._active_disruptions) > 0

        # ── PANIC ORDER CHECK ─────────────────────────────────────────────────
        # Gamma used without any active disruption = panic order.
        # Return immediately with ONLY the penalty — no positive bonuses can
        # offset it. This makes the negative signal unambiguous for RL training.
        if supplier_id == "gamma" and not disruption_active:
            return {
                "order_id": order_id,
                "sku": sku,
                "supplier_id": supplier_id,
                "requested_quantity": quantity,
                "delivered_quantity": delivered_qty,
                "price_per_unit": price_per_unit,
                "total_cost": round(total_cost, 2),
                "eta_day": arrival_day,
                "cash_remaining": round(self._cash_balance, 2),
                "reason_logged": action.reason,
                "warning": "Panic order penalty applied — Gamma used without active disruption.",
            }, REWARD_PANIC_ORDER_PENALTY   # always -0.3, nothing added

        # ── NORMAL ORDER REWARDS ──────────────────────────────────────────────
        # Only reached when supplier is alpha or beta, or gamma during disruption.
        reward = 0.0

        # Proactive bonus: still had enough days of stock when ordering
        avg_demand = self._demand_model.average_daily_demand(sku)
        days_of_stock = self._inventory.get(sku, 0) / max(1, avg_demand)
        lead_time = quotes[supplier_id]["lead_time_days"]
        if days_of_stock >= lead_time:
            reward += REWARD_PROACTIVE_ORDER_BONUS

        # Disruption pre-emption: heavily stocked before disruption hit
        if disruption_active and self._inventory.get(sku, 0) > 2 * self._safety_stock.get(sku, 0):
            reward += REWARD_DISRUPTION_PREEMPTED_BONUS

        # Reliability bonus: chose alpha (most reliable) or best available
        # Gamma NEVER gets this bonus — even during a disruption it's a compromise
        if supplier_id == "alpha":
            reward += REWARD_SUPPLIER_RELIABILITY_BONUS
        elif supplier_id == "beta" and "alpha" not in self._cfg["active_suppliers"]:
            reward += REWARD_SUPPLIER_RELIABILITY_BONUS

        return {
            "order_id": order_id,
            "sku": sku,
            "supplier_id": supplier_id,
            "requested_quantity": quantity,
            "delivered_quantity": delivered_qty,
            "price_per_unit": price_per_unit,
            "total_cost": round(total_cost, 2),
            "eta_day": arrival_day,
            "cash_remaining": round(self._cash_balance, 2),
            "reason_logged": action.reason,
        }, reward

    def _tool_cancel_order(self, order_id: Optional[str], reason: Optional[str]) -> Dict:
        for i, order in enumerate(self._pending_orders):
            if order["order_id"] == order_id:
                if order["eta_day"] - self._day > 2:
                    refund = order["total_cost"]
                    self._cash_balance += refund
                    self._month_spend -= refund
                    self._total_procurement_cost -= refund
                    self._pending_orders.pop(i)
                    return {"status": "cancelled", "order_id": order_id, "refund": refund, "reason_logged": reason}
                else:
                    return {"error": f"Order {order_id} is already in transit and cannot be cancelled."}
        return {"error": f"Order {order_id} not found."}

    def _tool_adjust_reorder_point(self, sku: Optional[str], new_threshold: Optional[int], reason: Optional[str]) -> Dict:
        if not sku or sku not in self._active_skus:
            return {"error": f"Invalid or inactive SKU: {sku}"}
        if new_threshold is None or new_threshold < 0:
            return {"error": "new_threshold must be a non-negative integer."}
        old = self._reorder_points.get(sku, 0)
        self._reorder_points[sku] = new_threshold
        return {"sku": sku, "old_reorder_point": old, "new_reorder_point": new_threshold, "reason_logged": reason}

    def _tool_adjust_safety_stock(self, sku: Optional[str], new_level: Optional[int], reason: Optional[str]) -> Dict:
        if not sku or sku not in self._active_skus:
            return {"error": f"Invalid or inactive SKU: {sku}"}
        if new_level is None or new_level < 0:
            return {"error": "new_level must be a non-negative integer."}
        old = self._safety_stock.get(sku, 0)
        self._safety_stock[sku] = new_level
        return {"sku": sku, "old_safety_stock": old, "new_safety_stock": new_level, "reason_logged": reason}

    def _tool_check_warehouse_capacity(self) -> Dict:
        total = self._total_inventory()
        capacity = self._cfg["warehouse_capacity"]
        return {
            "warehouse_capacity": capacity,
            "units_in_use": total,
            "units_available": capacity - total,
            "utilization_pct": round(100 * total / capacity, 1),
            "in_transit_units": sum(o["delivered_quantity"] for o in self._pending_orders),
        }

    def _tool_expedite_order(self, order_id: Optional[str], reason: Optional[str]):
        for order in self._pending_orders:
            if order["order_id"] == order_id:
                if order["eta_day"] - self._day <= 1:
                    return {"error": f"Order {order_id} already arrives tomorrow."}, 0.0
                premium = order["total_cost"] * 0.20
                if self._cash_balance < premium:
                    return {"error": f"Insufficient cash to expedite. Required: {premium:.0f} INR."}, 0.0
                self._cash_balance -= premium
                self._month_spend += premium
                old_eta = order["eta_day"]
                order["eta_day"] = max(self._day + 1, order["eta_day"] - 3)
                return {
                    "order_id": order_id,
                    "old_eta_day": old_eta,
                    "new_eta_day": order["eta_day"],
                    "expedite_fee": round(premium, 2),
                    "reason_logged": reason,
                }, 0.0
        return {"error": f"Order {order_id} not found."}, 0.0

    def _tool_transfer_between_skus(self, from_sku: Optional[str], to_sku: Optional[str], quantity: Optional[int]) -> Dict:
        if not from_sku or from_sku not in self._active_skus:
            return {"error": f"Invalid source SKU: {from_sku}"}
        if not to_sku or to_sku not in self._active_skus:
            return {"error": f"Invalid destination SKU: {to_sku}"}
        if not quantity or quantity <= 0:
            return {"error": "quantity must be a positive integer."}
        available = self._inventory.get(from_sku, 0)
        if quantity > available:
            return {"error": f"Only {available} units of {from_sku} available to transfer."}
        self._inventory[from_sku] -= quantity
        self._inventory[to_sku] = self._inventory.get(to_sku, 0) + quantity
        return {
            "from_sku": from_sku,
            "to_sku": to_sku,
            "units_transferred": quantity,
            "from_sku_remaining": self._inventory[from_sku],
            "to_sku_new_level": self._inventory[to_sku],
        }

    def _tool_observe_disruptions(self) -> Dict:
        if not self._active_disruptions:
            return {"disruption_active": False, "message": "No active supply disruptions."}
        return {
            "disruption_active": True,
            "disruptions": self._active_disruptions,
            "upcoming_schedule_hint": f"Episode may have up to {self._cfg['disruptions']} total disruptions.",
        }

    # ── Disruption management ─────────────────────────────────────────────────

    def _build_disruption_schedule(self) -> List[Dict]:
        n = self._cfg.get("disruptions", 0)
        if not n:
            return []
        schedule = []
        duration_days = self._cfg["duration_days"]
        window_start, window_end = 15, duration_days - 15
        suppliers = self._cfg["active_suppliers"]
        skus = self._active_skus

        for i in range(n):
            start_day = self._rng.randint(window_start + i * 15, min(window_end, window_start + (i + 1) * 20))
            duration = self._rng.randint(5, 12)
            supplier = self._rng.choice([s for s in suppliers if s != "alpha"])
            sku = self._rng.choice(skus)
            severity = round(self._rng.uniform(0.25, 0.55), 2)
            schedule.append({
                "start_day": start_day,
                "end_day": start_day + duration,
                "supplier_id": supplier,
                "affected_sku": sku,
                "severity": severity,
                "duration": duration,
            })
        return schedule

    def _process_disruption_schedule(self) -> None:
        for event in self._disruption_schedule:
            if event["start_day"] == self._day:
                self._supplier_model.apply_disruption(event["supplier_id"], event["severity"])
                disruption_info = {**event, "status": "active"}
                self._active_disruptions.append(disruption_info)
                self._episode_disruptions.append(disruption_info)
            if event["end_day"] == self._day:
                self._supplier_model.recover_disruption(event["supplier_id"])
                self._active_disruptions = [
                    d for d in self._active_disruptions
                    if d["supplier_id"] != event["supplier_id"]
                ]

    # ── New SKU launch ────────────────────────────────────────────────────────

    def _launch_new_sku(self, sku: str) -> None:
        """Add a new SKU to the active list with zero demand history."""
        if sku not in self._active_skus:
            self._active_skus.append(sku)
            self._inventory[sku] = 0
            self._reorder_points[sku] = SKU_CONFIG[sku]["base_demand"] * 5
            self._safety_stock[sku] = SKU_CONFIG[sku]["base_demand"]
            self._stockout_history[sku] = 0
            self._total_expired[sku] = 0
            self._daily_demand[sku] = []
            self._daily_filled[sku] = []
            self._daily_inventory[sku] = []
            self._demand_model._demand_history[sku] = []
            self._demand_model._active_skus.append(sku)

    # ── Graders ───────────────────────────────────────────────────────────────

    def _grade(self) -> float:
        if self._task_id == "easy":
            return self._grade_task_1()
        elif self._task_id == "medium":
            return self._grade_task_2()
        else:
            return self._grade_task_3()

    def _grade_task_1(self) -> float:
        """Score a single-SKU inventory control episode. Returns float in [0.0, 1.0]."""
        score = 0.0
        sku = "SKU-C"

        total_demand = sum(self._daily_demand.get(sku, [1]))
        total_filled = sum(self._daily_filled.get(sku, [0]))
        fill_rate = total_filled / max(1, total_demand)
        score += fill_rate * 0.40

        inv_levels = self._daily_inventory.get(sku, [0])
        avg_inventory = mean(inv_levels) if inv_levels else 0
        target = 1.5 * self._safety_stock.get(sku, 1)
        deviation = abs(avg_inventory - target) / max(1, target)
        score += max(0.0, 1.0 - deviation) * 0.30

        timing_scores = []
        avg_demand = self._demand_model.average_daily_demand(sku)
        lead_time = SKU_CONFIG[sku]["lead_time_max"]
        safety_buffer = 3
        # Use order_history instead of pending_orders to see all orders ever placed
        for order in [o for o in self._order_history if o["sku"] == sku]:
            order_day = order["order_day"]
            # Look up inventory level on the day the order was placed
            # Index is order_day - 1 because list is 0-indexed
            try:
                inv_at_order = self._daily_inventory[sku][order_day - 1]
            except (KeyError, IndexError):
                inv_at_order = 0
                
            days_stock = inv_at_order / max(1, avg_demand)
            ideal = lead_time + safety_buffer
            err = abs(days_stock - ideal)
            timing_scores.append(max(0.0, 1.0 - err / max(1, ideal)))
        score += (mean(timing_scores) if timing_scores else 0.5) * 0.20

        theoretical_min = (
            total_demand * SKU_CONFIG[sku]["unit_cost"]
            + self._safety_stock.get(sku, 0) * SKU_CONFIG[sku]["holding_cost_per_unit_per_day"] * self._cfg["duration_days"]
        )
        actual_cost = self._total_procurement_cost + self._total_carrying_cost
        cost_efficiency = min(theoretical_min / max(1, actual_cost), 1.0)
        score += cost_efficiency * 0.10

        score = min(max(score, 0.01), 0.99)
        return round(score, 4)

    def _grade_task_2(self) -> float:
        """Score a multi-SKU, multi-supplier episode with one disruption. Returns float in [0.0, 1.0]."""
        score = 0.0
        SKU_WEIGHTS = {"SKU-A": 0.25, "SKU-B": 0.30, "SKU-C": 0.15, "SKU-D": 0.20, "SKU-E": 0.10}

        weighted_fill = sum(
            SKU_WEIGHTS.get(sku, 0) * self._compute_fill_rate(sku)
            for sku in self._active_skus
            if sku in SKU_WEIGHTS
        )
        score += weighted_fill * 0.35

        if self._episode_disruptions:
            disr = self._episode_disruptions[0]
            start, end = disr["start_day"], disr["end_day"]
            affected = disr["affected_sku"]
            window = range(max(1, start - 3), min(len(self._daily_inventory.get(affected, [])), end + 5))
            inv = self._daily_inventory.get(affected, [])
            stockouts = sum(1 for d in window if d < len(inv) and inv[d] == 0)
            disruption_score = max(0.0, 1.0 - stockouts / max(1, len(list(window))))
            score += disruption_score * 0.25
        else:
            score += 0.25

        oracle_cost = self._compute_oracle_cost()
        actual_cost = self._total_procurement_cost
        score += min(oracle_cost / max(1, actual_cost), 1.0) * 0.20

        util_scores = []
        for u in self._daily_warehouse_util:
            if 0.60 <= u <= 0.85:
                util_scores.append(1.0)
            elif u < 0.60:
                util_scores.append(u / 0.60)
            else:
                util_scores.append(max(0.0, 1.0 - (u - 0.85) / 0.15))
        score += (mean(util_scores) if util_scores else 0.5) * 0.12

        sku_d_purchased = sum(
            o["requested_quantity"] for o in self._order_history if o["sku"] == "SKU-D"
        )
        sku_d_expired = self._total_expired.get("SKU-D", 0)
        if sku_d_purchased > 0:
            waste_rate = sku_d_expired / sku_d_purchased
            score += max(0.0, 1.0 - waste_rate * 3) * 0.08
        else:
            score += 0.08

        score = min(max(score, 0.01), 0.99)
        return round(score, 4)

    def _grade_task_3(self) -> float:
        """Score a full-quarter supply chain management episode. Returns float in [0.0, 1.0]. Seven components."""
        score = 0.0
        SKU_REVENUE_WEIGHTS = {
            "SKU-A": 0.20, "SKU-B": 0.35, "SKU-C": 0.10,
            "SKU-D": 0.15, "SKU-E": 0.12, "SKU-F": 0.08,
        }

        service_level = sum(
            SKU_REVENUE_WEIGHTS.get(sku, 0) * self._compute_fill_rate(sku)
            for sku in self._active_skus
            if sku in SKU_REVENUE_WEIGHTS
        )
        score += service_level * 0.30

        benchmark_cost = self._compute_benchmark_cost()
        actual_cost = self._total_procurement_cost + self._total_carrying_cost + self._total_stockout_cost
        score += min(benchmark_cost / max(1, actual_cost), 1.0) * 0.25

        surge = self._cfg.get("surge_days")
        if surge:
            s_start, s_end, surge_skus, _ = surge
            surge_demand_total = 0
            surge_filled_total = 0
            for sku in surge_skus:
                d_list = self._daily_demand.get(sku, [])
                f_list = self._daily_filled.get(sku, [])
                for day_idx in range(s_start - 1, min(s_end, len(d_list))):
                    surge_demand_total += d_list[day_idx]
                    surge_filled_total += f_list[day_idx]
            surge_score = surge_filled_total / max(1, surge_demand_total)
            score += surge_score * 0.15
        else:
            score += 0.15

        monthly_limit = self._cfg.get("monthly_cash_limit", float("inf"))
        cash_health = []
        for bal in self._daily_cash_balance:
            min_healthy = monthly_limit * 0.20 if monthly_limit != float("inf") else 0
            if bal >= min_healthy:
                cash_health.append(1.0)
            elif bal > 0:
                cash_health.append(bal / max(1, min_healthy))
            else:
                cash_health.append(0.0)
        score += (mean(cash_health) if cash_health else 1.0) * 0.12

        if "SKU-F" in self._active_skus:
            f_demand = sum(self._daily_demand.get("SKU-F", [0]))
            f_filled = sum(self._daily_filled.get("SKU-F", [0]))
            sku_f_fill = f_filled / max(1, f_demand)
            score += min(sku_f_fill / 0.75, 1.0) * 0.08
        else:
            score += 0.08

        if self._episode_disruptions:
            disruption_scores = []
            for disr in self._episode_disruptions:
                affected = disr["affected_sku"]
                inv = self._daily_inventory.get(affected, [])
                dem = self._daily_demand.get(affected, [])
                fil = self._daily_filled.get(affected, [])
                window = range(disr["start_day"] - 1, min(disr["end_day"] + 3, len(inv)))
                fills = [
                    fil[d] / max(1, dem[d])
                    for d in window
                    if d < len(fil) and d < len(dem)
                ]
                disruption_scores.append(mean(fills) if fills else 0.5)
            score += mean(disruption_scores) * 0.07
        else:
            score += 0.07

        sku_d_expired = self._total_expired.get("SKU-D", 0)
        sku_d_purchased = sum(
            o["requested_quantity"] for o in self._order_history if o["sku"] == "SKU-D"
        )
        if sku_d_purchased > 0:
            waste_rate = sku_d_expired / sku_d_purchased
            score += max(0.0, 1.0 - waste_rate * 3) * 0.03
        else:
            score += 0.03

        score = min(max(score, 0.01), 0.99)
        return round(score, 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _total_inventory(self) -> int:
        return sum(self._inventory.values())

    def _compute_fill_rate(self, sku: str) -> float:
        demand = self._daily_demand.get(sku, [])
        filled = self._daily_filled.get(sku, [])
        if not demand:
            return 1.0
        return sum(filled) / max(1, sum(demand))

    def _compute_oracle_cost(self) -> float:
        """Theoretical minimum procurement cost using cheapest available supplier."""
        total = 0.0
        for sku in self._active_skus:
            demand = sum(self._daily_demand.get(sku, []))
            cheapest_price = min(
                SKU_CONFIG[sku]["unit_cost"] * SUPPLIER_CONFIG[s]["price_multiplier"]
                for s in self._cfg["active_suppliers"]
            )
            total += demand * cheapest_price
        return total

    def _compute_benchmark_cost(self) -> float:
        """95% service level cost benchmark for the hard task."""
        total = 0.0
        for sku in self._active_skus:
            demand = sum(self._daily_demand.get(sku, []))
            best_price = min(
                SKU_CONFIG[sku]["unit_cost"] * SUPPLIER_CONFIG[s]["price_multiplier"]
                for s in self._cfg["active_suppliers"]
            )
            holding = self._safety_stock.get(sku, 0) * SKU_CONFIG[sku]["holding_cost_per_unit_per_day"] * self._cfg["duration_days"]
            total += demand * best_price * 0.95 + holding
        return total

    def _build_observation(
        self,
        tool_name: str,
        tool_result: Dict,
        reward: float = 0.0,
        grade_score: Optional[float] = None,
    ) -> SupplyChainObservation:
        """Package the current simulation state into a SupplyChainObservation."""
        disruption_active = len(self._active_disruptions) > 0
        disruption_details = self._active_disruptions[0] if disruption_active else {}

        fill_rates = {sku: self._compute_fill_rate(sku) for sku in self._active_skus}
        service_level = mean(fill_rates.values()) if fill_rates else 0.0
        util = self._total_inventory() / max(1, self._cfg["warehouse_capacity"])

        return SupplyChainObservation(
            tool_result=tool_result,
            tool_name_called=tool_name,
            current_day=self._day,
            inventory=dict(self._inventory),
            in_transit={
                sku: [o for o in self._pending_orders if o["sku"] == sku]
                for sku in self._active_skus
            },
            reorder_points=dict(self._reorder_points),
            safety_stock=dict(self._safety_stock),
            warehouse_capacity=self._cfg["warehouse_capacity"],
            warehouse_utilization=round(util, 3),
            cash_balance=round(self._cash_balance, 2),
            carrying_cost_today=round(
                sum(
                    self._inventory.get(sku, 0) * SKU_CONFIG[sku]["holding_cost_per_unit_per_day"]
                    for sku in self._active_skus
                ), 2
            ),
            total_carrying_cost=round(self._total_carrying_cost, 2),
            total_stockout_cost=round(self._total_stockout_cost, 2),
            total_procurement_cost=round(self._total_procurement_cost, 2),
            fill_rate=fill_rates,
            service_level=round(service_level, 4),
            stockout_history=dict(self._stockout_history),
            pending_orders=list(self._pending_orders),
            disruption_active=disruption_active,
            disruption_details=disruption_details,
            task_id=self._task_id,
            episode_day=self._day,
            done=self._done,
            reward=reward,
            grade_score=grade_score,
        )