"""
Data models for the Supply Chain RL Environment.

The supply_chain_env environment simulates a multi-product, multi-supplier
supply chain over a simulated quarter (90 days). An LLM-based RL agent
manages inventory, places purchase orders, selects suppliers, and responds
to disruptions — exactly as a real supply chain manager would.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, List, Any


class SupplyChainAction(Action):
    """
    Action for the Supply Chain environment.

    The agent issues one MCP tool call per step. The tool_name selects which of the 12 available tools to invoke; the remaining fields are the parameters for that tool. Fields unused by the chosen tool are ignored by the environment.
    """

    tool_name: str = Field(
        ...,
        description=(
            "Name of the MCP tool to invoke. One of: observe_inventory, "
            "get_demand_forecast, get_demand_history, get_supplier_quotes, "
            "place_order, cancel_order, adjust_reorder_point, adjust_safety_stock, "
            "check_warehouse_capacity, expedite_order, transfer_between_skus, "
            "observe_disruptions."
        ),
    )
    sku: Optional[str] = Field(
        default = None,
        description="Target SKU indentifier (e.g. 'SKU-A').Required by most tools.",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        description = "Supplier identifier ('alpha', 'beta', or 'gamma'). Used by place_order.",
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Number of units. Used by place_order, get_supplier_quotes, transfer_between_skus.",
    )
    order_id: Optional[str] = Field(
        default=None,
        description="Purchase order identifier. Used by cancel_order and expedite_order.",
    )
    new_threshold: Optional[int] = Field(
        default=None,
        description="New reorder point level (units). Used by adjust_reorder_point.",
    )
    new_level: Optional[int] = Field(
        default=None,
        description="New safety stock level (units). Used by adjust_safety_stock.",
    )
    horizon_days: Optional[int] = Field(
        default=14,
        description="Forecast horizon in days. Used by get_demand_forecast.",
    )
    lookback_days: Optional[int] = Field(
        default=30,
        description="History window in days. Used by get_demand_history.",
    )
    from_sku: Optional[str] = Field(
        default=None,
        description="Source SKU for warehouse space transfer. Used by transfer_between_skus.",
    )
    to_sku: Optional[str] = Field(
        default=None,
        description="Destination SKU for warehouse space transfer. Used by transfer_between_skus.",
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Agent's free-text justification for the action. "
            "Logged for analysis; does not affect simulation outcomes."
        ),
    )


class SupplyChainObservation(Observation):
    """
    Observation returned by the Supply Chain environment after every step.

    Contains the complete visible state of the supply chain, the tool
    result from the action just taken, the step reward, and episode
    termination flag.
    """

    # ── Tool result ───────────────────────────────────────────────────────────
    tool_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON result returned by the MCP tool that was just called.",
    )
    tool_name_called: str = Field(
        default="",
        description="Name of the tool that produced this observation.",
    )

    # ── Time ─────────────────────────────────────────────────────────────────
    current_day: int = Field(
        default=1,
        description="Current simulation day (1-indexed). Episode ends at the task duration.",
    )

    # ── Inventory snapshot ────────────────────────────────────────────────────
    inventory: Dict[str, int] = Field(
        default_factory=dict,
        description="Current on-hand stock level per SKU (units).",
    )
    in_transit: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Orders placed but not yet arrived, keyed by SKU. Each entry has quantity and eta_day.",
    )
    reorder_points: Dict[str, int] = Field(
        default_factory=dict,
        description="Reorder-point threshold per SKU (units). Agent can adjust these.",
    )
    safety_stock: Dict[str, int] = Field(
        default_factory=dict,
        description="Minimum buffer stock per SKU (units). Agent can adjust these.",
    )

    # ── Warehouse ─────────────────────────────────────────────────────────────
    warehouse_capacity: int = Field(
        default=10000,
        description="Total warehouse capacity across all SKUs (units).",
    )
    warehouse_utilization: float = Field(
        default=0.0,
        description="Fraction of warehouse capacity currently in use (0.0–1.0).",
    )

    # ── Financials ────────────────────────────────────────────────────────────
    cash_balance: float = Field(
        default=0.0,
        description="Available cash for purchasing inventory (INR).",
    )
    carrying_cost_today: float = Field(
        default=0.0,
        description="Total holding cost charged today across all SKUs (INR).",
    )
    total_carrying_cost: float = Field(
        default=0.0,
        description="Cumulative holding cost for the episode so far (INR).",
    )
    total_stockout_cost: float = Field(
        default=0.0,
        description="Cumulative lost-sales cost for the episode so far (INR).",
    )
    total_procurement_cost: float = Field(
        default=0.0,
        description="Cumulative procurement spend for the episode so far (INR).",
    )

    # ── Performance metrics ───────────────────────────────────────────────────
    fill_rate: Dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of demand met without delay per SKU (0.0–1.0).",
    )
    service_level: float = Field(
        default=0.0,
        description="Rolling 30-day average fill rate across all SKUs.",
    )
    stockout_history: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of days each SKU has been out of stock this episode.",
    )

    # ── Orders ────────────────────────────────────────────────────────────────
    pending_orders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Open purchase orders with order_id, sku, quantity, supplier, and eta_day.",
    )

    # ── Disruptions ───────────────────────────────────────────────────────────
    disruption_active: bool = Field(
        default=False,
        description="Whether a supply disruption is currently active.",
    )
    disruption_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details of the active disruption: affected_supplier, severity, eta_recovery_day.",
    )

    # ── Episode info ──────────────────────────────────────────────────────────
    task_id: str = Field(
        default="easy",
        description="Active task identifier: 'easy', 'medium', or 'hard'.",
    )
    episode_day: int = Field(
        default=1,
        description="Alias for current_day; provided for clarity in training logs.",
    )
    grade_score: Optional[float] = Field(
        default=None,
        description="Final grader score (0.0–1.0). Populated only when done=True.",
    )

