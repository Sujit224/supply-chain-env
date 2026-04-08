"""
Inference Script — Supply Chain RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    OPENAI_API_KEY OpenAI-compatible API key.
    LOCAL_IMAGE_NAME     The name of the local Docker image for the environment
                   (used with from_docker_image())

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the ROOT directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=easy env=supply_chain_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=(observe_inventory) reward=0.00 done=false error=null
    [STEP] step=2 action=(place_order,SKU-C,alpha,500) reward=0.50 done=false error=null
    [END] success=true steps=2 score=0.724 rewards=0.00,0.50

RESOURCE CONSTRAINTS
    - Runtime:  < 20 minutes total
    - Hardware: 2 vCPU, 8 GB RAM
    - MAX_STEPS is set conservatively to stay well within the time budget.
      Each LLM call ~ 2-4 s -> 60 steps x 3 s = ~3 min per task safely.
"""

import asyncio
import json
import math
import os
import subprocess
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from models import SupplyChainAction
from client import SupplyChainEnv
from server.mcp_tools import SUPPLY_CHAIN_TOOLS

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
    or os.environ.get("HF_TOKEN")
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("SUPPLY_CHAIN_TASK")
SINGLE_TASK = os.getenv("SUPPLY_CHAIN_SINGLE_TASK")
BENCHMARK = os.getenv("SUPPLY_CHAIN_BENCHMARK", "supply_chain_env")
SEED         = int(os.getenv("SUPPLY_CHAIN_SEED",  "42"))

# Resource-aware step budget — stays well within 20 min on remote endpoints
_STEP_BUDGETS = {"easy": 30, "medium": 60, "hard": 90}
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")

TEMPERATURE = 0.7
MAX_TOKENS  = 256
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STAGNANT_OBSERVATION_STEPS = 2
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

OBSERVATION_TOOLS = {
    "observe_inventory",
    "get_demand_forecast",
    "get_demand_history",
    "get_supplier_quotes",
    "check_warehouse_capacity",
    "observe_disruptions",
}

SKU_UNIT_COST = {
    "SKU-A": 120,
    "SKU-B": 4500,
    "SKU-C": 800,
    "SKU-D": 60,
    "SKU-E": 2200,
    "SKU-F": 1500,
}

# ── FIX 1: System prompt now tells agent to act, not just observe ─────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a supply chain manager. Your goal: maximise fill rate, minimise costs.

    IMPORTANT — only call ONE tool per step. Do NOT repeat the same tool consecutively
    unless the situation has changed. After observing, act — do not just observe again.

    Recommended flow each day:
      Step 1: observe_inventory         (check stock levels)
      Step 2: get_demand_forecast       (check upcoming demand)
      Step 3: place_order if needed     (order ONLY if on_hand < reorder_point)
      Step 4: observe_disruptions       (check for supply issues)
      Repeat next day.

    Ordering rules:
    - Only order if on_hand + in_transit < reorder_point + safety_stock.
    - Order quantity = (reorder_point + safety_stock) - on_hand - in_transit.
    - NEVER order if warehouse is above 85% full.
    - Use Supplier Alpha always (cheapest, most reliable).
    - Always include a 'reason' when placing orders.

    DO NOT call get_supplier_quotes or check_warehouse_capacity every single step —
    only call them when you are about to place an order.
""").strip()


# ── Mandatory stdout logging ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt helpers ────────────────────────────────────────────────────────────

# ── FIX 2: format_observation now diagnoses state and tells agent what to do ──

def format_observation(obs: Any, step: int) -> str:
    """
    Build a concise user prompt from the current observation.
    Explicitly tells the agent what to do next based on state —
    reduces the chance of looping on the same tool repeatedly.
    """
    lines = [
        f"=== Day {obs.current_day} | Step {step} ===",
        f"Cash: {obs.cash_balance:,.0f} INR | Warehouse: {obs.warehouse_utilization:.0%} used"
        f" | Service level: {obs.service_level:.0%} | Disruption: {obs.disruption_active}",
    ]

    if obs.disruption_active and obs.disruption_details:
        d = obs.disruption_details
        lines.append(
            f"DISRUPTION ACTIVE: supplier={d.get('supplier_id')} "
            f"severity={d.get('severity')} ends Day {d.get('end_day')}"
        )

    lines.append("\nInventory status:")
    needs_order = []
    for sku, qty in obs.inventory.items():
        fr  = obs.fill_rate.get(sku, 0)
        rp  = obs.reorder_points.get(sku, 0)
        ss  = obs.safety_stock.get(sku, 0)
        in_transit = sum(
            o["delivered_quantity"] for o in obs.pending_orders if o["sku"] == sku
        )
        total_available = qty + in_transit
        status = "LOW - ORDER NOW" if total_available < rp else "OK"
        lines.append(
            f"  {sku}: on_hand={qty} in_transit={in_transit} total={total_available}"
            f" | reorder_pt={rp} safety={ss} | fill={fr:.0%} | {status}"
        )
        if total_available < rp:
            order_qty = max(0, (rp + ss) - total_available)
            needs_order.append(f"{sku} needs ~{order_qty} units")

    if needs_order:
        lines.append(f"\nACTION NEEDED: {', '.join(needs_order)}")
        lines.append("-> Call place_order now. Use supplier=alpha.")
    else:
        lines.append("\nAll SKUs sufficiently stocked. Observe or adjust policy if needed.")

    lines.append(f"\nLast tool: {obs.tool_name_called}")
    lines.append(f"Last result: {json.dumps(obs.tool_result)[:250]}")

    return "\n".join(lines)


def action_to_str(action: SupplyChainAction) -> str:
    """Function-like action string for [STEP] logs."""
    args: List[str] = []
    for field in [
        "sku", "supplier_id", "quantity", "order_id",
        "new_threshold", "new_level", "horizon_days",
        "lookback_days", "from_sku", "to_sku", "reason",
    ]:
        val = getattr(action, field, None)
        if val is None:
            continue
        if isinstance(val, str):
            args.append(f"{field}={val!r}")
        else:
            args.append(f"{field}={val}")
    return f"{action.tool_name}({', '.join(args)})"


def extract_step_reward(result: Any) -> float:
    """
    Extract step reward, prioritizing the squashed reward from the observation.
    """
    # 1. Look for the squashed reward inside the observation FIRST
    if hasattr(result, "observation") and hasattr(result.observation, "reward"):
        if getattr(result.observation, "reward") is not None:
            return float(result.observation.reward)
            
    # 2. Fallback to the top-level result reward
    reward = getattr(result, "reward", None)
    if reward is not None:
        return float(reward)
        
    # 3. Final fallback
    return 0.01


def normalize_and_clamp_reward(raw_reward: float) -> float:
    """
    Normalize reward to (0, 1) then clamp to (0.01, 0.99).

    - If reward is already in (0, 1), keep scale.
    - If reward appears unbounded (outside [0, 1] or non-finite), squash with sigmoid.
    - Finally clamp for benchmark output requirements.
    """
    if not math.isfinite(raw_reward):
        normalized = 1.0 if raw_reward > 0 else 0.0
    elif 0.0 <= raw_reward <= 1.0:
        normalized = raw_reward
    else:
        try:
            normalized = 1.0 / (1.0 + math.exp(-raw_reward))
        except OverflowError:
            normalized = 0.0 if raw_reward < 0 else 1.0

    return min(max(normalized, 0.01), 0.99)


def finalize_score(result: Any, rewards: List[float]) -> float:
    """
    Ensure emitted task score is always strictly within (0, 1).

    Priority:
    1) Use grader score when available.
    2) Fallback to mean normalized rewards for incomplete episodes.
    3) Final safety fallback to 0.01.
    """
    grade_score = None
    if result is not None and hasattr(result, "observation"):
        grade_score = getattr(result.observation, "grade_score", None)

    if grade_score is not None:
        return min(max(float(grade_score), 0.01), 0.99)

    if rewards:
        avg_reward = sum(rewards) / max(1, len(rewards))
        return min(max(float(avg_reward), 0.01), 0.99)

    return 0.01


def build_progress_action(obs: Any, avoid_orders: bool = False) -> SupplyChainAction:
    """
    Build a low-risk non-observation action to break read-only loops.
    """
    if avoid_orders:
        fallback_sku = next(iter(obs.inventory.keys()), "SKU-C")
        current_rp = int(obs.reorder_points.get(fallback_sku, 0))
        lowered_rp = max(0, int(current_rp * 0.95))
        return SupplyChainAction(
            tool_name="adjust_reorder_point",
            sku=fallback_sku,
            new_threshold=lowered_rp,
            reason="Auto-progress: temporary order cooldown after cash-limit rejection.",
        )

    available_space = max(0, int(obs.warehouse_capacity) - int(sum(obs.inventory.values())))

    candidates: List[tuple[float, str, int]] = []
    for sku, on_hand in obs.inventory.items():
        rp = int(obs.reorder_points.get(sku, 0))
        ss = int(obs.safety_stock.get(sku, 0))
        in_transit = sum(
            int(o.get("delivered_quantity", 0))
            for o in obs.pending_orders
            if o.get("sku") == sku
        )
        target = rp + ss
        shortfall = target - (on_hand + in_transit)
        if shortfall <= 0 or available_space <= 0:
            continue

        qty = min(shortfall, available_space)
        if qty <= 0:
            continue

        unit_cost = SKU_UNIT_COST.get(sku, 1000)
        estimated_cost = qty * unit_cost
        urgency = shortfall / max(1, rp)
        # Prefer urgent + cheaper replenishments to avoid cash-limit failures.
        score = urgency / max(1.0, estimated_cost / 100000.0)
        candidates.append((score, sku, qty))

    if candidates:
        _, sku, qty = max(candidates, key=lambda x: x[0])
        return SupplyChainAction(
            tool_name="place_order",
            sku=sku,
            supplier_id="alpha",
            quantity=qty,
            reason="Auto-progress: break observation loop with cost-aware replenishment.",
        )

    # Fallback: always use a safe non-observation action that advances time
    # without triggering warehouse overflow penalties.
    fallback_sku = next(iter(obs.inventory.keys()), "SKU-C")
    current_rp = int(obs.reorder_points.get(fallback_sku, 0))
    return SupplyChainAction(
        tool_name="adjust_reorder_point",
        sku=fallback_sku,
        new_threshold=current_rp,
        reason="Auto-progress: break observation loop with no-op threshold refresh.",
    )


def parse_tool_call(response: Any) -> SupplyChainAction:
    """
    Parse the first tool call from the OpenAI response into a SupplyChainAction.
    Falls back to observe_inventory if the model sends no tool call.
    """
    choice = response.choices[0]
    if not choice.message.tool_calls:
        return SupplyChainAction(tool_name="observe_inventory")

    tc = choice.message.tool_calls[0]
    try:
        args: Dict[str, Any] = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        args = {}

    return SupplyChainAction(
        tool_name=tc.function.name,
        sku=args.get("sku"),
        supplier_id=args.get("supplier_id"),
        quantity=args.get("quantity"),
        order_id=args.get("order_id"),
        new_threshold=args.get("new_threshold"),
        new_level=args.get("new_level"),
        horizon_days=args.get("horizon_days"),
        lookback_days=args.get("lookback_days"),
        from_sku=args.get("from_sku"),
        to_sku=args.get("to_sku"),
        reason=args.get("reason"),
    )


def get_agent_action(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    obs: Any,
    step: int,
) -> tuple[SupplyChainAction, List[Dict[str, Any]]]:
    """
    Call the LLM with the current observation and return the chosen action.
    Appends the assistant turn to messages so the model has full history.
    """
    messages.append({
        "role": "user",
        "content": format_observation(obs, step),
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=SUPPLY_CHAIN_TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        assistant_msg = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        if response.choices[0].message.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in response.choices[0].message.tool_calls
            ]
        messages.append(assistant_msg)
        action = parse_tool_call(response)

        # Feed the tool result back into history so model sees its own outputs
        if response.choices[0].message.tool_calls:
            tc_id = response.choices[0].message.tool_calls[0].id
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(obs.tool_result),
            })

    except Exception:
        action = SupplyChainAction(tool_name="observe_inventory")

    return action, messages


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    env = None
    client = None

    # 18-min hard timeout — leaves 2-min buffer before the 20-min limit
    TIMEOUT_SECONDS = 18 * 60
    # Submission default: always benchmark all three tasks.
    # Use SUPPLY_CHAIN_SINGLE_TASK only for local debugging.
    tasks_to_run = [SINGLE_TASK] if SINGLE_TASK else ["easy", "medium", "hard"]

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        if LOCAL_IMAGE_NAME:
            env = await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = SupplyChainEnv(base_url="http://localhost:8000")
    except Exception:
        if env is None:
            env = SupplyChainEnv(base_url="http://localhost:8000")
    
    for task_id in tasks_to_run:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        stagnant_observation_steps = 0
        order_cooldown_steps = 0

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await asyncio.wait_for(
                env.reset(task_id=task_id, seed=SEED),
                timeout=60,
            )

            start_time = asyncio.get_event_loop().time()
            task_max_steps = int(MAX_STEPS_OVERRIDE) if MAX_STEPS_OVERRIDE else _STEP_BUDGETS.get(task_id, 130)

            for step in range(1, task_max_steps + 1):
                if result.done:
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > TIMEOUT_SECONDS:
                    break

                try:
                    obs = result.observation
                    action, messages = get_agent_action(client, messages, obs, step)

                    chosen_tool = (action.tool_name or "").strip().lower()
                    if chosen_tool in OBSERVATION_TOOLS:
                        stagnant_observation_steps += 1
                    else:
                        stagnant_observation_steps = 0

                    if stagnant_observation_steps >= MAX_STAGNANT_OBSERVATION_STEPS:
                        action = build_progress_action(obs, avoid_orders=(order_cooldown_steps > 0))
                        stagnant_observation_steps = 0

                    result = await asyncio.wait_for(env.step(action), timeout=30)
                except Exception:
                    # Robust fallback: keep trajectory moving even if model/tool call fails.
                    obs = result.observation
                    action = build_progress_action(obs, avoid_orders=(order_cooldown_steps > 0))
                    result = await asyncio.wait_for(env.step(action), timeout=30)
                    stagnant_observation_steps = 0

                reward = extract_step_reward(result)
                done = result.done
                steps_taken = step
                rewards.append(reward)

                error_msg = None
                new_obs = result.observation
                if isinstance(new_obs.tool_result, dict):
                    error_msg = new_obs.tool_result.get("error")
                    if error_msg:
                        error_msg = str(error_msg).replace("\n", " ").replace("\r", " ")

                if error_msg and "monthly cash limit exceeded" in error_msg.lower():
                    order_cooldown_steps = 3
                elif order_cooldown_steps > 0:
                    order_cooldown_steps -= 1

                log_step(
                    step=step,
                    action=action_to_str(action),
                    reward=reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

            score = finalize_score(result, rewards)
            # Final safety check to keep score in (0, 1) per benchmark rules
            score = min(max(score, 0.01), 0.99)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception:
            pass
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    try:
        if env is not None:
            await env.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())