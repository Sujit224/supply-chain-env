# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP tool definitions for the Supply Chain RL Environment.

Each of the 12 tools is declared as an OpenAI-compatible tool schema.
These schemas are served at GET /schema so that any LLM client
(Claude, GPT-4o, etc.) can discover and call them dynamically.

The agent chains multiple tool calls within a single decision step,
building a multi-step reasoning chain before committing to actions —
exactly as a real supply chain manager would use SAP or Oracle SCM.
"""

from typing import Any, Dict, List

# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────

SUPPLY_CHAIN_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "observe_inventory",
            "description": (
                "Get current stock levels, in-transit orders, reorder points, and "
                "safety stock for one SKU or all active SKUs. "
                "Call this first on every decision step for situational awareness."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": (
                            "SKU identifier (e.g. 'SKU-A'). "
                            "Omit to retrieve inventory for all active SKUs."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demand_forecast",
            "description": (
                "Retrieve a 1–14 day ahead demand forecast with noise for one or all SKUs. "
                "The forecast is deliberately noisy — it reflects the true signal plus "
                "Gaussian noise, mirroring real forecasting tools. "
                "Use this to decide how much to order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier. Omit for forecasts across all active SKUs.",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of future days to forecast (1–14). Default: 14.",
                        "minimum": 1,
                        "maximum": 14,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demand_history",
            "description": (
                "Get actual recorded demand over the past N days for one or all SKUs. "
                "Use this to validate forecast accuracy and identify demand patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier. Omit for history across all active SKUs.",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Number of past days to return (default: 30).",
                        "minimum": 1,
                        "maximum": 90,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_supplier_quotes",
            "description": (
                "Get price, lead time, reliability, and MOQ quotes from all available "
                "suppliers for a given SKU and quantity. "
                "Always call this before placing a purchase order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier. Omit for quotes across all active SKUs.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Intended order quantity in units.",
                        "minimum": 1,
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": (
                "Place a purchase order for a SKU with a specific supplier. "
                "The order arrives after the supplier's lead time (stochastic). "
                "Core replenishment action — use after checking inventory and forecasts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier (e.g. 'SKU-A').",
                    },
                    "supplier_id": {
                        "type": "string",
                        "description": "Supplier identifier: 'alpha', 'beta', or 'gamma'.",
                        "enum": ["alpha", "beta", "gamma"],
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of units to order. Must meet supplier MOQ.",
                        "minimum": 1,
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Free-text justification for this order. "
                            "Logged for analysis; does not affect outcomes. "
                            "Example: 'SKU-D inventory below safety stock; Supplier Beta disrupted; "
                            "using Gamma despite premium to prevent stockout within 1 day.'"
                        ),
                    },
                },
                "required": ["sku", "supplier_id", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": (
                "Cancel a pending purchase order that has not yet shipped. "
                "A full refund is issued for orders more than 2 days from arrival. "
                "Use when demand drops unexpectedly and excess inventory is a risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Purchase order identifier (e.g. 'PO-0012').",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Justification for cancellation.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_reorder_point",
            "description": (
                "Update the inventory threshold that triggers a reorder review for a SKU. "
                "Proactive policy tuning — raise this during disruptions or high-demand "
                "periods, lower it when demand is stable and lead times are short."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier.",
                    },
                    "new_threshold": {
                        "type": "integer",
                        "description": "New reorder point in units (≥ 0).",
                        "minimum": 0,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Justification for the change.",
                    },
                },
                "required": ["sku", "new_threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_safety_stock",
            "description": (
                "Update the minimum buffer stock to maintain for a SKU. "
                "Raise safety stock during active disruptions or high-demand volatility. "
                "Lower it for stable, predictable SKUs to reduce carrying costs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "SKU identifier.",
                    },
                    "new_level": {
                        "type": "integer",
                        "description": "New safety stock level in units (≥ 0).",
                        "minimum": 0,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Justification for the change.",
                    },
                },
                "required": ["sku", "new_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_warehouse_capacity",
            "description": (
                "Get total warehouse capacity, current utilization, and available space. "
                "Call this before placing large orders to avoid overflow rejection."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expedite_order",
            "description": (
                "Pay a 20 % premium to rush an in-transit order and reduce its ETA by "
                "up to 3 days. Use only in emergencies — stockout imminent, no other "
                "supplier available, and the cost is justified by the stockout penalty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Purchase order identifier to expedite.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Emergency justification.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_between_skus",
            "description": (
                "Reassign warehouse space (physical units) from one SKU to another. "
                "Use to rebalance capacity allocation when one SKU is overstocked "
                "and another is critically low."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_sku": {
                        "type": "string",
                        "description": "Source SKU to transfer units from.",
                    },
                    "to_sku": {
                        "type": "string",
                        "description": "Destination SKU to receive the units.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of units to transfer.",
                        "minimum": 1,
                    },
                },
                "required": ["from_sku", "to_sku", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "observe_disruptions",
            "description": (
                "Check for active supply disruptions, affected suppliers, severity, "
                "and estimated recovery timeline. "
                "Call this daily for situational awareness — disruptions directly "
                "affect supplier reliability and lead times."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def get_tool_schema() -> Dict[str, Any]:
    """
    Return the complete tool schema for serving at GET /schema.

    Returns:
        Dict with tool list and action/observation field descriptions.
    """
    return {
        "tools": SUPPLY_CHAIN_TOOLS,
        "action_fields": {
            "tool_name": "string — name of the MCP tool to call",
            "sku": "string | null — SKU identifier (e.g. 'SKU-A')",
            "supplier_id": "string | null — 'alpha', 'beta', or 'gamma'",
            "quantity": "int | null — units to order / transfer",
            "order_id": "string | null — purchase order identifier",
            "new_threshold": "int | null — new reorder point (units)",
            "new_level": "int | null — new safety stock (units)",
            "horizon_days": "int | null — forecast horizon (1–14)",
            "lookback_days": "int | null — history window (1–90)",
            "from_sku": "string | null — source SKU for transfer",
            "to_sku": "string | null — destination SKU for transfer",
            "reason": "string | null — free-text justification (logged only)",
        },
    }
