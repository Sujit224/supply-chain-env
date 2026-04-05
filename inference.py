"""
Inference Script — Supply Chain RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local Docker image for the environment
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
      Each LLM call ~ 2–4 s → 60 steps × 3 s = ~3 min per task safely.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from supply_chain_env import SupplyChainAction, SupplyChainEnv
from supply_chain_env.server.mcp_tools import SUPPLY_CHAIN_TOOLS

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY    = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("SUPPLY_CHAIN_TASK",      "easy")
BENCHMARK    = os.getenv("SUPPLY_CHAIN_BENCHMARK", "supply_chain_env")
SEED         = int(os.getenv("SUPPLY_CHAIN_SEED",  "42"))

# ── Resource-aware step budget ────────────────────────────────────────────────
# Runtime limit: 20 min. Each LLM call takes ~2–4 s on a remote endpoint.
# easy=30d, medium=60d, hard=90d — agent needs ~2 tool calls per day.
# Budget per task: easy=70, medium=130, hard=190 — all fit in ~10 min worst case.
_STEP_BUDGETS = {"easy": 70, "medium": 130, "hard": 190}
MAX_STEPS = int(os.getenv("MAX_STEPS", str(_STEP_BUDGETS.get(TASK_NAME, 130))))

TEMPERATURE = 0.7
MAX_TOKENS  = 256   # kept small to reduce latency on 2-vCPU inference servers
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a supply chain manager. Maximise service level, minimise costs.
    Use the 12 MCP tools each step. Decision order every step:
    1. observe_inventory  2. observe_disruptions  3. get_demand_forecast
    4. get_supplier_quotes  5. place_order (if needed)

    Rules:
    - Prefer Supplier Alpha (cheap, reliable). Use Gamma only in emergencies.
    - Keep warehouse 60-85% full. Never go cash-negative.
    - SKU-D is perishable — order small quantities frequently.
    - During disruptions: raise safety stock, switch suppliers immediately.
    - Always set 'reason' field when placing orders.
    """
).strip()


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt helpers ────────────────────────────────────────────────────────────

def format_observation(obs: Any, step: int) -> str:
    """
    Format SupplyChainObservation into a concise user prompt for the LLM.
    Kept intentionally short to reduce token usage on memory-constrained hardware.
    """
    lines = [
        f"Day {obs.current_day} | Step {step} | Cash: {obs.cash_balance:,.0f} INR"
        f" | Warehouse: {obs.warehouse_utilization:.0%} | SvcLvl: {obs.service_level:.0%}"
        f" | Disruption: {obs.disruption_active}",
    ]

    if obs.disruption_active and obs.disruption_details:
        d = obs.disruption_details
        lines.append(
            f"DISRUPTION: supplier={d.get('supplier_id')} "
            f"severity={d.get('severity')} ends Day {d.get('end_day')}"
        )

    # One line per SKU — all critical info on a single row
    for sku, qty in obs.inventory.items():
        fr  = obs.fill_rate.get(sku, 0)
        rp  = obs.reorder_points.get(sku, 0)
        ss  = obs.safety_stock.get(sku, 0)
        int = [o for o in obs.pending_orders if o["sku"] == sku]
        in_transit_qty = sum(o["delivered_quantity"] for o in int)
        lines.append(
            f"{sku}: on_hand={qty} in_transit={in_transit_qty}"
            f" fill={fr:.0%} reorder={rp} safety={ss}"
        )

    # Last tool result — capped at 300 chars to prevent prompt bloat
    lines.append(f"Last: {obs.tool_name_called} -> {json.dumps(obs.tool_result)[:300]}")

    return "\n".join(lines)


def action_to_str(action: SupplyChainAction) -> str:
    """Compact one-line action string used in [STEP] log lines."""
    parts = [action.tool_name]
    for field in ["sku", "supplier_id", "quantity", "order_id"]:
        val = getattr(action, field, None)
        if val is not None:
            parts.append(str(val))
    return "(" + ",".join(parts) + ")"


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

        messages.append(response.choices[0].message.model_dump())
        action = parse_tool_call(response)

        # Append tool result so the model sees its own call history next step
        if response.choices[0].message.tool_calls:
            tc_id = response.choices[0].message.tool_calls[0].id
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(obs.tool_result),
            })

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        action = SupplyChainAction(tool_name="observe_inventory")

    return action, messages


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await SupplyChainEnv.from_docker_image(IMAGE_NAME)
    else:
        env = SupplyChainEnv(base_url="http://localhost:8000")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards:  List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    # Hard wall-clock guard: abort gracefully before the 20-min runtime limit
    TIMEOUT_SECONDS = 18 * 60   # 18 min — leaves 2 min buffer for teardown

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await asyncio.wait_for(
            env.reset(task_id=TASK_NAME, seed=SEED),
            timeout=60,
        )

        start_time = asyncio.get_event_loop().time()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Abort if we are approaching the 20-min wall-clock limit
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                print(f"[DEBUG] Timeout guard triggered at step {step} ({elapsed:.0f}s elapsed)", flush=True)
                break

            obs = result.observation
            action, messages = get_agent_action(client, messages, obs, step)

            result = await asyncio.wait_for(env.step(action), timeout=30)

            reward = result.reward or 0.0
            done   = result.done
            error  = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score   = result.observation.grade_score or 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except asyncio.TimeoutError:
        print("[DEBUG] Hard timeout hit — forcing episode end", flush=True)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())