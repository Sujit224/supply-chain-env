---
title: Supply Chain Env
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Supply Chain RL Environment

An OpenEnv-compliant MCP Reinforcement Learning Environment for multi-product
supply chain management — inventory control, supplier selection, and demand forecasting.

## Overview

This environment trains an LLM-based RL agent to manage a multi-product,
multi-supplier supply chain over a simulated quarter (90 days). The agent
must forecast demand, place purchase orders, select suppliers, manage warehouse
capacity, and respond to disruptions — exactly as a real supply chain manager would.

## Tasks

| Task | Duration | SKUs | Suppliers | Disruptions | Difficulty |
|------|----------|------|-----------|-------------|------------|
| `easy` | 30 days | 1 (SKU-C) | 1 (Alpha) | None | Easy |
| `medium` | 60 days | 5 | 3 | 1 surprise | Medium |
| `hard` | 90 days | 6 | 3 | 2–3 + surge + cash limit | Hard |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export SUPPLY_CHAIN_TASK="easy"   # easy | medium | hard
```

### 3. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run inference

```bash
python inference.py
```

## Docker

### Build

```bash
docker build -t supply_chain_env .
```

### Run server

```bash
docker run -p 8000:8000 supply_chain_env
```

### Run inference against Docker

```bash
export LOCAL_IMAGE_NAME=supply_chain_env
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python inference.py
```

## Project Structure

```
supply_chain_env/
├── inference.py              ← entry point (root, as required)
├── models.py                 ← Pydantic Action + Observation models
├── client.py                 ← EnvClient for training code
├── openenv.yaml              ← OpenEnv spec
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── server/
│   ├── app.py                ← FastAPI server
│   ├── environment.py        ← core simulation + graders
│   ├── demand_model.py       ← stochastic demand generator
│   ├── supplier_model.py     ← supplier reliability + pricing
│   └── mcp_tools.py          ← 12 MCP tool definitions
└── baseline/
    └── inference.py          ← multi-task baseline script
```

## MCP Tools (Action Space)

| Tool | Purpose |
|------|---------|
| `observe_inventory` | Get stock levels, in-transit orders, reorder points |
| `get_demand_forecast` | 14-day ahead noisy demand forecast |
| `get_demand_history` | Last N days of actual demand |
| `get_supplier_quotes` | Price, lead time, reliability from all suppliers |
| `place_order` | Place a purchase order with a supplier |
| `cancel_order` | Cancel a pending order not yet shipped |
| `adjust_reorder_point` | Update the reorder trigger threshold |
| `adjust_safety_stock` | Update the minimum buffer stock |
| `check_warehouse_capacity` | Get available warehouse space |
| `expedite_order` | Rush an in-transit order (20% fee) |
| `transfer_between_skus` | Reassign warehouse space between SKUs |
| `observe_disruptions` | Check active supply disruptions |

## Scoring

All graders are fully deterministic (pure arithmetic on episode logs — no LLM judge).

| Task | Components | Key metrics |
|------|-----------|-------------|
| Easy | 4 | Fill rate, inventory efficiency, order timing, cost |
| Medium | 5 | Weighted fill rate, disruption response, supplier cost, warehouse util, perishable waste |
| Hard | 7 | Service level, total cost, surge handling, cash management, new SKU ramp-up, disruption resilience, perishable waste |

## Expected Baseline Scores (GPT-4o zero-shot)

| Agent | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| Oracle (perfect info) | 1.00 | 1.00 | 1.00 | 1.00 |
| GPT-4o zero-shot | ~0.72 | ~0.58 | ~0.41 | ~0.57 |
| Simple reorder-point heuristic | ~0.64 | ~0.42 | ~0.28 | ~0.45 |
| Random policy | ~0.21 | ~0.18 | ~0.12 | ~0.17 |
| RL-trained agent (target) | >0.85 | >0.75 | >0.62 | >0.74 |