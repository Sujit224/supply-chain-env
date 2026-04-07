"""
Inference Script — Supply Chain RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
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
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("SUPPLY_CHAIN_TASK", "easy")
BENCHMARK = os.getenv("SUPPLY_CHAIN_BENCHMARK", "supply_chain_env")
SEED         = int(os.getenv("SUPPLY_CHAIN_SEED",  "42"))

# Resource-aware step budget — stays well within 20 min on remote endpoints
_STEP_BUDGETS = {"easy": 70, "medium": 130, "hard": 190}
MAX_STEPS = int(os.getenv("MAX_STEPS", str(_STEP_BUDGETS.get(TASK_NAME, 130))))

TEMPERATURE = 0.7
MAX_TOKENS  = 256
SUCCESS_SCORE_THRESHOLD = 0.5

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

        # Feed the tool result back into history so model sees its own outputs
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
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards:  List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False
    env = None
    client = None

    # 18-min hard timeout — leaves 2-min buffer before the 20-min limit
    TIMEOUT_SECONDS = 18 * 60

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        if LOCAL_IMAGE_NAME:
            try:
                # Check if image exists, build if it doesn't
                inspect_result = subprocess.run(
                    ["docker", "image", "inspect", LOCAL_IMAGE_NAME],
                    capture_output=True,
                    text=True,
                )
                if inspect_result.returncode != 0:
                    print(f"[DEBUG] Image '{LOCAL_IMAGE_NAME}' not found locally. Building...", flush=True)
                    subprocess.run(["docker", "build", "-t", LOCAL_IMAGE_NAME, "."], check=True)
                    print(f"[DEBUG] Successfully built '{LOCAL_IMAGE_NAME}'.", flush=True)
            except Exception as build_ex:
                print(f"[DEBUG] Auto-build failed: {build_ex}", flush=True)

            env = await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = SupplyChainEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"[DEBUG] Setup phase exception: {e}. Falling back to localhost:8000 if env not initialized.", flush=True)
        if env is None:
            env = SupplyChainEnv(base_url="http://localhost:8000")
            
    try:
        result = await asyncio.wait_for(
            env.reset(task_id=TASK_NAME, seed=SEED),
            timeout=60,
        )

        start_time = asyncio.get_event_loop().time()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                print(f"[DEBUG] Timeout guard at step {step} ({elapsed:.0f}s)", flush=True)
                break

            obs = result.observation
            action, messages = get_agent_action(client, messages, obs, step)

            result = await asyncio.wait_for(env.step(action), timeout=30)

            reward      = result.reward or 0.0
            done        = result.done
            steps_taken = step
            rewards.append(reward)

            error_msg = None
            if isinstance(obs.tool_result, dict):
                error_msg = obs.tool_result.get("error")
                if error_msg:
                    error_msg = str(error_msg).replace("\n", " ").replace("\r", " ")

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        score   = result.observation.grade_score if (result and hasattr(result.observation, 'grade_score') and result.observation.grade_score is not None) else 0.01
        score   = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except asyncio.TimeoutError:
        print("[DEBUG] Hard timeout — forcing episode end", flush=True)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass

        # Exactly ONE [END] line — env.close() errors are silenced above
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())