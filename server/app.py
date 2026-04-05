"""
FastAPI application for the Supply Chain RL Environment.

This module creates an HTTP server that exposes SupplyChainEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment (supports task_id and seed params)
    - POST /step: Execute a SupplyChainAction (MCP tool call)
    - GET  /state: Get current environment state
    - GET  /schema: Get action / observation schemas + MCP tool definitions
    - WS   /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SupplyChainAction, SupplyChainObservation
    from .environment import SupplyChainEnvironment
except ModuleNotFoundError:
    from models import SupplyChainAction, SupplyChainObservation
    from server.environment import SupplyChainEnvironment


# Create the app — increase max_concurrent_envs for parallel training runs
app = create_app(
    SupplyChainEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="supply_chain_env",
    max_concurrent_envs=4,   # supports parallel training across easy/medium/hard
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m supply_chain_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments with multiple parallel environments:
        uvicorn supply_chain_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supply Chain RL Environment server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
