---
title: Emergency First-Response Decision Engine
sdk: docker
app_port: 7860
tags:
  - openenv
  - fastapi
  - reinforcement-learning
  - decision-support
---

# Emergency First-Response Decision Engine

## Environment Overview

The Emergency First-Response Decision Engine is a deterministic OpenEnv-compatible environment for evaluating AI agents that support non-expert first responders during real emergencies. It models the kind of stepwise reasoning a bystander, security guard, or family member may need when facing cardiac arrest, severe bleeding, or a roadside injury before professional help arrives.

This is a real-world workflow rather than a game. The agent must decide what to do next using partial observations, changing patient state, and safety constraints. The environment is designed for both learning and evaluation: it exposes typed actions and observations, dense reward signals, deterministic graders, and reproducible task trajectories.

The repository now also includes a proper working reinforcement learning agent based on tabular Q-learning. The RL agent trains only through the environment's `reset()` and `step()` interface and can persist a learned policy to disk for later evaluation.

## Real-World Motivation

First-response decision support is a practical agent benchmark because:

- interventions must happen in the right order
- some actions are beneficial only in the correct clinical context
- delays and unsafe behavior have meaningful consequences
- observations are partial and improve only after targeted assessment

This makes the environment useful for agent evaluation, planning research, policy learning, and safety-focused benchmarking.

## OpenEnv Interface

The core environment class is [environment/env.py](C:\Users\admin\Desktop\New Folder\environment\env.py) and implements:

- `reset(task_id: str | None = None) -> Observation`
- `step(action: Action) -> tuple[Observation, float, bool, dict]`
- `state() -> InternalState`

Typed models are defined in [environment/models.py](C:\Users\admin\Desktop\New Folder\environment\models.py):

- `Observation`
- `Action`
- `RewardSignal`
- `StepOutput`
- `InternalState`

The OpenEnv metadata file is [openenv.yaml](C:\Users\admin\Desktop\New Folder\openenv.yaml).

## Action Space

The environment uses a discrete, structured action space:

- `CALL_EMERGENCY`: Contact emergency medical services.
- `CHECK_SCENE_SAFETY`: Assess hazards before committing to patient contact.
- `CHECK_RESPONSIVENESS`: Determine whether the patient can respond.
- `CHECK_BREATHING`: Assess breathing status.
- `CHECK_PULSE`: Assess pulse status.
- `START_CPR`: Begin cardiopulmonary resuscitation when indicated.
- `USE_AED`: Apply a defibrillator in cardiac arrest when appropriate.
- `APPLY_PRESSURE`: Control severe external bleeding with direct pressure.
- `CONTROL_AIRWAY`: Support a compromised airway.
- `PLACE_RECOVERY_POSITION`: Protect the airway of an unconscious but breathing patient when safe.
- `MONITOR_PATIENT`: Reassess the patient after key interventions.
- `WAIT`: Take no action for one step.

## Observation Space

Each observation includes:

- `task_id`
- `difficulty`
- `scenario_summary`
- `patient_condition`
- `time_elapsed`
- `actions_taken`
- `environment_context`
- `available_actions`
- `last_action_effect`
- `risk_level`

The `patient_condition` object includes:

- `conscious_status`
- `breathing_status`
- `bleeding_severity`
- `pulse_status`
- `airway_status`

Observations are intentionally partial. Some fields remain `unknown` until the agent performs relevant assessment actions.

## Reward Design

Rewards are dense and meaningful across the full episode:

- positive reward for clinically appropriate life-saving actions
- positive reward for correct sequencing
- positive reward for stabilizing progression
- negative reward for repeated actions without reassessment
- negative reward for delays in urgent interventions
- negative reward for clearly unsafe or irrelevant actions
- terminal bonus for successful stabilization
- terminal penalty for critical failure

The `info` dictionary returned by `step()` includes a typed reward breakdown via `reward_signal`, making it easy for judges to inspect how each step was scored.

## RL Agent

The trainable RL agent is implemented in [rl_agent.py](C:\Users\admin\Desktop\New Folder\rl_agent.py).

Properties:

- learns only from `reset()` and `step()`
- uses a deterministic state encoder over structured observations
- trains with tabular Q-learning
- uses a fixed random seed for reproducibility
- saves the learned Q-table to `artifacts/q_table.json`
- can be evaluated greedily after training

Training entrypoint:

- [train_rl.py](C:\Users\admin\Desktop\New Folder\train_rl.py)

Run:

```bash
python train_rl.py
```

This produces:

- a saved learned policy in `artifacts/q_table.json`
- per-task training summary
- per-task evaluation summary

## Tasks

Task definitions live in [environment/tasks.py](C:\Users\admin\Desktop\New Folder\environment\tasks.py). Deterministic graders live in [environment/grader.py](C:\Users\admin\Desktop\New Folder\environment\grader.py).

### Easy: `cardiac_arrest_easy`

Scenario:
An adult collapses in an airport terminal with an AED nearby and clear evidence of non-normal breathing.

Goal:
Activate help quickly, confirm breathing status, start CPR, use the AED, and monitor the patient.

Optimal sequence:
`CALL_EMERGENCY -> CHECK_BREATHING -> START_CPR -> USE_AED -> MONITOR_PATIENT`

### Medium: `severe_bleeding_medium`

Scenario:
A kitchen worker has a deep forearm laceration with rapid blood loss and visible environmental hazards.

Goal:
Manage hazards, contact emergency help, control hemorrhage, reassess circulation, and monitor for shock.

Optimal sequence:
`CHECK_SCENE_SAFETY -> CALL_EMERGENCY -> APPLY_PRESSURE -> CHECK_PULSE -> MONITOR_PATIENT`

### Hard: `road_accident_hard`

Scenario:
A motorcyclist lies beside a road with moving traffic, heavy thigh bleeding, shallow breathing, and evolving airway compromise.

Goal:
Prioritize scene safety, call for help, control severe bleeding, reassess breathing, manage the airway, reassess circulation, and monitor the patient as the situation evolves.

Optimal sequence:
`CHECK_SCENE_SAFETY -> CALL_EMERGENCY -> APPLY_PRESSURE -> CHECK_BREATHING -> CONTROL_AIRWAY -> CHECK_PULSE -> MONITOR_PATIENT`

## Graders

Each task has a deterministic grader that returns a score between `0.0` and `1.0` based on:

- action correctness
- sequence alignment
- efficiency
- harmful or redundant behavior

The same action sequence always yields the same score.

## API Endpoints

The FastAPI service in [app.py](C:\Users\admin\Desktop\New Folder\app.py) exposes:

- `GET /` : single-page UI when the frontend bundle exists
- `GET /healthz` : health check
- `GET /tasks` : task metadata
- `GET /state` : current internal environment state
- `POST /reset` : start a task episode
- `POST /step` : advance the environment one action
- `GET /docs` : Swagger UI

## Simple UI

A minimal judge-friendly React UI is included in [frontend](C:\Users\admin\Desktop\New Folder\frontend). In the containerized deployment it is served by FastAPI from the root URL, so judges only need one running service.

Judge flow:

1. Open the root URL.
2. Click a task card.
3. Click `Start Selected Scenario`.
4. Click an action button.
5. Click `Submit Step`.
6. Click `Reset Current Scenario` to rerun the same task.

## Local Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Run the standalone frontend in development mode if desired:

```bash
cd frontend
npm install
npm run dev
```

## Docker Usage

Build:

```bash
docker build -t emergency-first-response-engine .
```

Run:

```bash
docker run --rm -p 7860:7860 emergency-first-response-engine
```

Then open:

- `http://localhost:7860`

## Hugging Face Spaces

This repository is prepared for a Docker Space deployment:

- README metadata declares `sdk: docker`
- the container serves the UI and API from one process
- the service binds to `${PORT:-7860}`
- the repo is tagged with `openenv`

## Running Inference

The baseline script is [inference.py](C:\Users\admin\Desktop\New Folder\inference.py).

Inference behavior:

- if `artifacts/q_table.json` exists, it uses the trained RL policy
- otherwise it falls back to the deterministic baseline policy
- if model credentials are available, the OpenAI client is initialized for compliant LLM-based policy experimentation
- the environment rollout itself remains deterministic and reproducible

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="your-hf-token"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:API_BASE_URL="https://your-llm-endpoint/v1"
$env:MODEL_NAME="gpt-4.1-mini"
$env:HF_TOKEN="your-hf-token"
```

Run:

```bash
python train_rl.py
python inference.py
```

Structured logs are emitted in the required format:

- `[START] task=...`
- `[STEP] step=... reward=...`
- `[END] task=...`

## Baseline Scores

Using the bundled deterministic fallback policy and the task-optimal trajectories:

- `cardiac_arrest_easy`: `1.0`
- `severe_bleeding_medium`: `1.0`
- `road_accident_hard`: `1.0`

Using the trained RL agent after `python train_rl.py`, the learned policy is expected to solve all three tasks consistently with high scores and deterministic replay.

## Judge Checklist

Before submission, verify:

- `docker build` succeeds
- `docker run` starts the app and `GET /healthz` returns `200`
- `POST /reset` and `POST /step` return valid JSON
- `GET /tasks` lists all 3 tasks
- `python train_rl.py` produces `artifacts/q_table.json`
- `python inference.py` completes within the runtime budget
- each grader returns values in `0.0` to `1.0`
- repeated runs with the same policy produce identical environment trajectories
