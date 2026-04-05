# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supply Chain RL Environment Client"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupplyChainAction, SupplyChainObservation


class SupplyChainEnv(
    EnvClient[SupplyChainAction,SupplyChainObservation,State]
):
    """
    Client for the Supply Chain RL Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    The agent interacts with the supply chain simulator exclusively through
    MCP tool calls (SupplyChainAction), matching how a real supply chain
    manager would use enterprise software such as SAP or Oracle SCM.

    Example:
        >>> with SupplyChainEnv(base_url="http://localhost:8000") as client:
        ...     # Start an easy episode
        ...     result = client.reset(task_id="easy", seed=42)
        ...     print(result.observation.current_day)
        ...
        ...     # Observe inventory
        ...     result = client.step(SupplyChainAction(
        ...         tool_name="observe_inventory",
        ...         sku="SKU-C",
        ...     ))
        ...     print(result.observation.tool_result)
        ...
        ...     # Place a replenishment order
        ...     result = client.step(SupplyChainAction(
        ...         tool_name="place_order",
        ...         sku="SKU-C",
        ...         supplier_id="alpha",
        ...         quantity=500,
        ...         reason="Below reorder point with 7-day lead time approaching",
        ...     ))
        ...     print(result.observation.cash_balance)

    Example with Docker:
        >>> client = SupplyChainEnv.from_docker_image("supply_chain_env:latest")
        >>> try:
        ...     result = client.reset(task_id="hard", seed=0)
        ...     while not result.observation.done:
        ...         action = my_agent.act(result.observation)
        ...         result = client.step(action)
        ...     print(f"Final score: {result.observation.grade_score}")
        ... finally:
        ...     client.close()
    """


    def _step_payload(self, action: SupplyChainAction) -> Dict[str , Any]:
        """
        Convert SupplyChainAction to JSON payload for the step message.

        Args:
            action: SupplyChainAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        payload: Dict[str,Any] = {"tool_name":action.tool_name}

        optional_fields = [
            "sku", "supplier_id", "quantity", "order_id",
            "new_threshold", "new_level", "horizon_days", "lookback_days",
            "from_sku", "to_sku", "reason",
        ]
        for field in optional_fields:
            value = getattr(action, field, None)
            if value is not None:
                payload[field] = value

        return payload


    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupplyChainObservation]:
        """
        Parse server response into StepResult[SupplyChainObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with SupplyChainObservation.
        """
        obs_data = payload.get("observation", {})

        observation = SupplyChainObservation(
            # Tool result
            tool_result=obs_data.get("tool_result", {}),
            tool_name_called=obs_data.get("tool_name_called", ""),
            # Time
            current_day=obs_data.get("current_day", 1),
            # Inventory snapshot
            inventory=obs_data.get("inventory", {}),
            in_transit=obs_data.get("in_transit", {}),
            reorder_points=obs_data.get("reorder_points", {}),
            safety_stock=obs_data.get("safety_stock", {}),
            # Warehouse
            warehouse_capacity=obs_data.get("warehouse_capacity", 10000),
            warehouse_utilization=obs_data.get("warehouse_utilization", 0.0),
            # Financials
            cash_balance=obs_data.get("cash_balance", 0.0),
            carrying_cost_today=obs_data.get("carrying_cost_today", 0.0),
            total_carrying_cost=obs_data.get("total_carrying_cost", 0.0),
            total_stockout_cost=obs_data.get("total_stockout_cost", 0.0),
            total_procurement_cost=obs_data.get("total_procurement_cost", 0.0),
            # Performance
            fill_rate=obs_data.get("fill_rate", {}),
            service_level=obs_data.get("service_level", 0.0),
            stockout_history=obs_data.get("stockout_history", {}),
            # Orders
            pending_orders=obs_data.get("pending_orders", []),
            # Disruptions
            disruption_active=obs_data.get("disruption_active", False),
            disruption_details=obs_data.get("disruption_details", {}),
            # Episode info
            task_id=obs_data.get("task_id", "easy"),
            episode_day=obs_data.get("episode_day", 1),
            grade_score=obs_data.get("grade_score"),
            # OpenEnv base fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )


    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
