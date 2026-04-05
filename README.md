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

> An OpenEnv-compatible environment for evaluating AI agents that assist non-expert first responders — bystanders, security guards, and family members — during cardiac arrest, severe bleeding, and roadside trauma.

---

## Overview

This is a real-world decision-support benchmark, not a game. The agent navigates genuine clinical workflows under time pressure, with partial observations that reveal only as much as the agent actively assesses. Patient condition deteriorates if critical actions are delayed. Every design choice — from the reward function to the observation gating — reflects how real emergencies actually unfold.

The environment is built for both learning and evaluation. It exposes:

- Typed actions and observations via Pydantic models
- Dense, shaped reward signals with clinical time-decay
- Deterministic graders with reproducible trajectories
- A trained tabular Q-learning agent as a baseline
- An LLM-backed inference agent using the OpenAI-compatible API

---

## Why This Domain

Emergency first response is an ideal agent benchmark because:

- interventions must happen in the correct clinical order
- some actions are only beneficial in the right context (e.g. USE_AED requires confirmed cardiac arrest)
- delays have real consequences — the reward function models survival-rate degradation over time
- observations are intentionally partial and improve only through targeted assessment actions
- the hard task requires multi-step reasoning that cannot be resolved from the initial observation alone

---

## OpenEnv Interface

**Core environment class:** `environment/env.py` — `EmergencyFirstResponseDecisionEngine`

| Method | Signature |
|--------|-----------|
| `reset` | `(task_id: str \| None = None) -> Observation` |
| `step` | `(action: Action) -> tuple[Observation, float, bool, dict]` |
| `state` | `() -> InternalState` |

**Typed models** in `environment/models.py`:

- `Observation` — what the agent sees each step
- `Action` — discrete action with typed `ActionType`
- `RewardSignal` — per-step reward breakdown (returned in `info`)
- `StepOutput` — full API response model
- `InternalState` — ground truth state (used by graders, not exposed to agent)

**OpenEnv metadata:** `openenv.yaml`

---

## Action Space

| Action | Description |
|--------|-------------|
| `CALL_EMERGENCY` | Contact emergency medical services |
| `CHECK_SCENE_SAFETY` | Assess hazards before approaching the patient |
| `CHECK_RESPONSIVENESS` | Determine whether the patient can respond |
| `CHECK_BREATHING` | Assess breathing status — reveals `breathing_status` and `airway_status` |
| `CHECK_PULSE` | Assess circulation — reveals `pulse_status` |
| `START_CPR` | Begin CPR when cardiac arrest is confirmed |
| `USE_AED` | Apply defibrillator when indicated |
| `APPLY_PRESSURE` | Control severe external bleeding |
| `CONTROL_AIRWAY` | Support a compromised airway |
| `PLACE_RECOVERY_POSITION` | Protect the airway of an unconscious but breathing patient |
| `MONITOR_PATIENT` | Reassess after key interventions |
| `WAIT` | Take no action for one step |

---

## Observation Space

Each observation contains:

| Field | Type | Notes |
|-------|------|-------|
| `task_id` | str | Always visible |
| `difficulty` | enum | Always visible |
| `scenario_summary` | str | Always visible |
| `time_elapsed` | int | Steps taken so far |
| `actions_taken` | list | History of actions |
| `available_actions` | list | Valid actions this step |
| `last_action_effect` | str | Narrative effect of previous action |
| `environment_context` | str | Hidden until `CHECK_SCENE_SAFETY` |
| `risk_level` | str | Hidden until `CHECK_SCENE_SAFETY` |
| `patient_condition.conscious_status` | enum | Hidden until `CHECK_RESPONSIVENESS` |
| `patient_condition.breathing_status` | enum | Hidden until `CHECK_BREATHING` |
| `patient_condition.airway_status` | enum | Hidden until `CHECK_BREATHING` |
| `patient_condition.pulse_status` | enum | Hidden until `CHECK_PULSE` |
| `patient_condition.bleeding_severity` | enum | Partially visible at reset for trauma tasks |

**Partial observability is enforced.** Fields return `"unknown"` until the corresponding assessment action is taken. The agent cannot infer clinical state from the initial observation — it must actively assess before acting.

---

## Reward Design

Rewards are dense and shaped across the full episode:

| Signal | Effect |
|--------|--------|
| Clinically appropriate action | Positive reward |
| Correct sequencing | Bonus reward |
| Patient stabilisation (terminal) | Large terminal bonus |
| Repeated actions without reassessment | Negative reward |
| Critical action delayed | Time-decay penalty (see below) |
| Unsafe or irrelevant action | Negative reward |
| Patient deterioration triggered | −0.15 per deterioration event |
| Episode failure (terminal) | Large terminal penalty |

### Intervention time-decay

Critical actions (CPR, APPLY_PRESSURE, CONTROL_AIRWAY, USE_AED) earn time-decayed rewards:

```
reward × max(0.5, 1.0 − 0.08 × steps_before_this_action)
```

CPR at step 1 earns full reward. CPR at step 5 earns 68%. CPR at step 8 earns 50%. This directly models the clinical reality that survival rates decline roughly 10% per minute without intervention.

### Patient deterioration

If the agent takes 3 or more consecutive non-critical actions, patient condition worsens:

- Cardiac tasks: `breathing_status` degrades one level
- Bleeding tasks: `bleeding_severity` worsens one level
- Road accident: both degrade

Each deterioration event emits −0.15 and is reported in `reward_signal.deterioration_penalty`. Optimal sequences always take a critical action within 2 steps, so deterioration never triggers on the optimal path.

The full per-step reward breakdown is available in the `info["reward_signal"]` field returned by `step()`.

---

## Tasks

Task definitions: `environment/tasks.py` — Graders: `environment/grader.py`

### `cardiac_arrest_easy`
**Scenario:** An adult collapses in an airport terminal. An AED is visible nearby. Breathing is clearly absent.

**Goal:** Call for help, confirm breathing status, start CPR, use the AED, monitor.

**Optimal sequence:**
```
CALL_EMERGENCY → CHECK_BREATHING → START_CPR → USE_AED → MONITOR_PATIENT
```

---

### `severe_bleeding_medium`
**Scenario:** A kitchen worker has a deep forearm laceration with rapid blood loss and visible environmental hazards.

**Goal:** Manage hazards, call for help, control haemorrhage, reassess circulation, monitor for shock.

**Optimal sequence:**
```
CHECK_SCENE_SAFETY → CALL_EMERGENCY → APPLY_PRESSURE → CHECK_PULSE → MONITOR_PATIENT
```

---

### `road_accident_hard`
**Scenario:** A motorcyclist lies beside a road with moving traffic, heavy thigh bleeding, shallow breathing, and evolving airway compromise. The scene is dynamic — patient state changes as the episode progresses.

**Goal:** Prioritise scene safety, call for help, control bleeding, reassess breathing, manage the airway, reassess circulation, monitor as the situation evolves.

**Optimal sequence:**
```
CHECK_SCENE_SAFETY → CALL_EMERGENCY → APPLY_PRESSURE → CHECK_BREATHING
→ CONTROL_AIRWAY → CHECK_PULSE → MONITOR_PATIENT
```

---

### `anaphylaxis_medium`
**Scenario:** A 28-year-old has a severe allergic reaction after eating at a restaurant. Hives, facial swelling, difficulty breathing, dropping blood pressure. Her EpiPen is in her bag.

**Goal:** Assess the environment, call for help, support breathing, manage the airway, reassess circulation, monitor.

**Optimal sequence:**
```
CHECK_SCENE_SAFETY → CALL_EMERGENCY → CHECK_BREATHING → CONTROL_AIRWAY
→ CHECK_PULSE → MONITOR_PATIENT
```

---

### `choking_easy`
**Scenario:** An elderly man at a restaurant is clutching his throat, cannot speak or cough, and his face is turning red. No breathing sounds are audible. Other diners are present.

**Goal:** Confirm responsiveness, call for help, clear the airway, confirm breathing, monitor.

**Optimal sequence:**
```
CHECK_RESPONSIVENESS → CALL_EMERGENCY → CONTROL_AIRWAY → CHECK_BREATHING → MONITOR_PATIENT
```

---

## Graders

Each task has a deterministic grader returning a score in `[0.0, 1.0]` based on:

- action correctness
- sequence alignment with the optimal path
- efficiency (step count relative to optimal)
- presence of harmful or redundant actions

The same action sequence always yields the same score. Graders operate on internal state values — not on the agent's (partial) observation — so partial observability does not affect grading accuracy.

Random or naive action sequences consistently score below 0.5. Optimal sequences score ≥ 0.95.

---

## RL Agent

Implemented in `rl_agent.py` — `QLearningEmergencyAgent`.

- trains exclusively via `reset()` and `step()` — no environment internals accessed
- uses a deterministic state encoder over structured observations
- tabular Q-learning with fixed random seed for reproducibility
- saves learned policy to `artifacts/q_table.json`
- greedy evaluation mode after training

**Train:**
```bash
python train_rl.py
```

Produces `artifacts/q_table.json`, per-task training summary, and per-task evaluation summary.

---

## Inference

The baseline inference script is `inference.py` (root level).

**Agent behaviour:**

1. The LLM is called at every step via `client.chat.completions.create()` — this is always the primary decision-maker.
2. The LLM receives the full structured observation and is prompted for chain-of-thought clinical reasoning before selecting an action:
```json
{"reasoning": "Patient shows absent breathing — CPR is the next critical step.", "action": "START_CPR"}
```
3. If the LLM returns an invalid or unparseable action, the Q-table is used as a fallback.
4. If no Q-table exists, the deterministic baseline policy is used.

**Required environment variables:**
```bash
export API_BASE_URL="https://your-llm-endpoint/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export OPENAI_API_KEY="your-openai-key"   # optional if HF_TOKEN is set
```

**Windows PowerShell:**
```powershell
$env:API_BASE_URL="https://your-llm-endpoint/v1"
$env:MODEL_NAME="your-model-name"
$env:HF_TOKEN="your-hf-token"
```

**Run:**
```bash
python train_rl.py   # optional — produces Q-table for fallback
python inference.py
```

**Structured log format:**
```
[START] task=cardiac_arrest_easy env=emergency_first_response_decision_engine model=gpt-4.1-mini
[STEP] step=1 action=CALL_EMERGENCY reward=0.30 done=false error=null
[STEP] step=2 action=CHECK_BREATHING reward=0.20 done=false error=null
[END] success=true steps=5 rewards=0.30,0.20,0.35,0.25,0.50
```

---

## API Endpoints

The FastAPI service in `app.py` exposes:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | React UI (served from frontend build) |
| `GET` | `/healthz` | Health check — returns `{"status": "ok"}` |
| `GET` | `/tasks` | All task metadata including optimal sequences |
| `GET` | `/state` | Current internal environment state |
| `GET` | `/trace` | Full action-observation-reward trace for current episode |
| `POST` | `/reset` | Start a task episode |
| `POST` | `/step` | Advance the environment one action |
| `GET` | `/docs` | Swagger UI |

Invalid actions submitted to `/step` return a structured error with a −0.2 penalty rather than a 500 response.

---

## Local Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Frontend (development mode):
```bash
cd frontend
npm install
npm run dev
```

---

## Docker

```bash
docker build -t emergency-first-response-engine .
docker run --rm -p 7860:7860 emergency-first-response-engine
```

Open `http://localhost:7860`

---

## Hugging Face Spaces

This repository is ready for Docker Space deployment:

- `sdk: docker` declared in README metadata
- single process serves both UI and API
- binds to `${PORT:-7860}`
- tagged with `openenv`

---

## Baseline Scores

| Task | Difficulty | Optimal score | Random baseline |
|------|------------|---------------|-----------------|
| `cardiac_arrest_easy` | Easy | 1.0 | < 0.3 |
| `severe_bleeding_medium` | Medium | 1.0 | < 0.3 |
| `road_accident_hard` | Hard | 0.97 | < 0.25 |
| `anaphylaxis_medium` | Medium | 1.0 | < 0.3 |
| `choking_easy` | Easy | 1.0 | < 0.3 |

---

## Pre-submission Checklist

- [ ] `python validate_submission.py` prints `validation_passed`
- [ ] `docker build` completes without errors
- [ ] `docker run` starts the app and `GET /healthz` returns `200`
- [ ] `POST /reset` and `POST /step` return valid JSON
- [ ] `GET /tasks` lists all 5 tasks
- [ ] `python train_rl.py` produces `artifacts/q_table.json`
- [ ] `python inference.py` completes within 20 minutes
- [ ] LLM is called at every step (confirm `client.chat.completions.create` appears in logs)
- [ ] Each grader returns values in `[0.0, 1.0]`
- [ ] Random action sequences score below 0.5 on all tasks
- [ ] Repeated runs with the same policy produce identical environment trajectories
