# OpenEnv Supply Chain Complete Development & Validation Plan

This document outlines the systematic strategy to build, test, and validate the `Supply Chain Env` for the OpenEnv evaluation challenge. Following this methodology ensures compliance with all automated, agentic, and human-evaluation criteria.

## 1. Project Structural Adjustments (Done)
- **Goal:** Adhere strictly to the submission constraints.
- **Action:** Flatten the workspace so `inference.py`, `openenv.yaml`, `Dockerfile`, and `README.md` all reside at the top-level repository root instead of an inner module directory. 
- **Validation:** `openenv validate` run from the root directory ensures `app: server.app:app` starts completely inside the correct context.

## 2. OpenEnv Interface & Requirements Completeness
- **Goal:** Ensure full spec alignment.
- **Tasks Definition:** Ensure `easy`, `medium`, and `hard` tasks successfully configure differing durations, SKU limits, and difficulty multipliers via `openenv.yaml` and `server/environment.py`. 
- **Grading & Rewards:** Verify that `step()` yields continuous partial rewards while parsing to bounded `[0.0, 1.0]` graded scores at episode termination.
- **Pydantic Types:** Action, observation, and state spaces correctly typed. 

## 3. Inference Script Implementation (`inference.py`)
- **Goal:** Implement the OpenAI client loop generating deterministic structured logs while honoring standard environment variables.
- **Action:** 
  - Expose default environmental fallback rules: `API_BASE_URL` and `MODEL_NAME`.
  - Fetch secret keys conditionally matching the rules: `os.getenv("HF_TOKEN") or os.getenv("API_KEY")`.
  - Standardize logging functions `log_start`, `log_step`, and `log_end` formatted exactly to specification standard with strict whitespace and precision matching.
  - Implement a `TIMEOUT_GUARD` bounding operations softly beneath the 20-minute threshold.

## 4. Dockerization & Deployment Validation
- **Goal:** Secure the multi-mode container execution target.
- **Action:**
  - Build `Dockerfile` testing dependencies and paths accurately (e.g., matching the `server/app.py` location).
  - Emulate the Hugging Face space launch via local startup parameters (`uvicorn server.app:app --host 0.0.0.0 --port 8000`).

## 5. Comprehensive Pre-Submission Checks
- Execute all commands found inside the provided validation script.
- **Check 1:** Mock Hugging Face space ping (`/reset`).
- **Check 2:** Confirm `docker build --tag supply_chain_env .` completes successfully.
- **Check 3:** Ensure `openenv validate` parses `openenv.yaml` unhandled errors.
- **Check 4:** Simulate one full inference run per task checking score compliance (`0.0 <= score <= 1.0`) and observing strict formatting properties.
