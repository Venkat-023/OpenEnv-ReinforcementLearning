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

OpenEnv-compatible reinforcement learning environment for emergency first-response decision-making under partial observability, time pressure, and clinically shaped rewards.

## What This Repo Does

This environment is designed to evaluate agents that must make medically plausible first-response decisions for:

- cardiac arrest
- severe external bleeding
- deceptive roadside trauma with hidden shock
- anaphylaxis
- choking

It includes:

- a FastAPI server in `app.py`
- an OpenEnv environment in `environment/env.py`
- deterministic task graders in `environment/grader.py`
- a tabular Q-learning baseline in `rl_agent.py`
- an LLM-first inference runner in `inference.py`
- a React frontend control panel in `frontend/`

## Core Design

The environment intentionally avoids trivial one-shot pattern matching:

- observations are partially hidden until the agent performs relevant checks
- critical delays cause deterioration
- critical interventions are time-decayed
- the hard trauma task hides the true diagnosis behind misleading initial cues
- reward feedback includes a clinical explanation for each step

## API Surface

The FastAPI app exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve frontend build if present |
| `GET` | `/healthz` | Health check |
| `GET` | `/state` | Current internal environment state |
| `GET` | `/tasks` | Task metadata |
| `POST` | `/reset` | Start or reset an episode |
| `POST` | `/step` | Advance one action |
| `GET` | `/docs` | Swagger UI |

`/healthz` still returns:

```json
{"status": "ok"}
```

## Observation Model

The environment returns an `Observation` with:

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

### Partial Observability

Some values remain `"unknown"` until the correct assessment action is taken.

Reveal rules:

- `CHECK_RESPONSIVENESS` reveals `conscious_status`
- `CHECK_BREATHING` reveals `breathing_status` and `airway_status`
- `CHECK_PULSE` reveals `pulse_status`
- `CHECK_SCENE_SAFETY` reveals `environment_context` and `risk_level`

Trauma/bleeding visibility:

- `bleeding_severity` is visible at reset only for `severe_bleeding_medium`
- the deceptive hard trauma task starts with bleeding hidden to avoid leaking the true condition too early

## Scenario-Aware Action Availability

The environment now returns scenario- and state-aware `available_actions` instead of always exposing the full action list.

This improves:

- frontend action panel relevance
- invalid action handling
- trajectory realism
- training signal quality

Examples:

- `USE_AED` only becomes available after CPR has started in the cardiac arrest task
- `CONTROL_AIRWAY` becomes available in airway-driven tasks after breathing assessment
- the deceptive hard trauma task opens circulation and stabilization actions only after assessment steps reveal enough context

Invalid submitted actions are handled safely and return a structured penalty instead of crashing the API.

## Reward Signal

Each `step()` response includes:

```json
{
  "reward_signal": {
    "action_reward": 1.0,
    "reason": "Checked breathing before intervention - aligns with ABC protocol",
    "time_decay_factor": 1.0,
    "total": -0.9
  }
}
```

Backward-compatible fields are still included in the reward payload:

- `value`
- `components`
- `rationale`

### Reward Meaning

- `action_reward`: direct reward or penalty for the submitted action
- `reason`: clinical explanation of why that action was rewarded or penalized
- `time_decay_factor`: intervention urgency multiplier for critical actions
- `total`: full step reward after base cost, progression, deterioration, and terminal effects

### Clinical Reasoning Strings

Reasons are rule-based, not LLM-generated.

Examples:

- `Checked breathing before intervention - aligns with ABC protocol`
- `Performed CPR without confirmed pulseless apnea - violates basic life support sequence`
- `USE_AED applied without confirmed cardiac arrest - contraindicated and potentially harmful`
- `Assessed circulation to look for occult shock - this is how hidden internal bleeding becomes apparent`

## Time Pressure and Deterioration

Critical actions:

- `CALL_EMERGENCY`
- `START_CPR`
- `APPLY_PRESSURE`
- `CONTROL_AIRWAY`
- `USE_AED`

If the agent takes 3 consecutive non-critical actions, the patient deteriorates.

Effects:

- cardiac tasks: breathing degrades
- bleeding tasks: bleeding worsens
- deceptive trauma task: both can worsen

Each deterioration event adds a penalty of `-0.15`.

## Time-Decay for Critical Interventions

Critical interventions are discounted using:

```text
max(0.5, 1.0 - 0.08 * steps_before_this_action)
```

This creates a medically meaningful reward surface:

- early intervention earns full value
- delayed intervention still helps, but less
- benefit bottoms out at 50%

## Tasks

Task definitions live in `environment/tasks.py`.

### `cardiac_arrest_easy`

- obvious arrest
- AED access nearby
- straightforward CPR/AED sequence

Optimal sequence:

```text
CALL_EMERGENCY -> CHECK_BREATHING -> START_CPR -> USE_AED -> MONITOR_PATIENT
```

### `severe_bleeding_medium`

- visible major hemorrhage
- scene hazard considerations
- requires bleeding control and circulation reassessment

Optimal sequence:

```text
CHECK_SCENE_SAFETY -> CALL_EMERGENCY -> APPLY_PRESSURE -> CHECK_PULSE -> MONITOR_PATIENT
```

### `road_accident_hard`

- deceptive trauma task
- minor visible bleeding masks evolving internal hemorrhagic shock
- true danger is not fully inferable from the opening description
- rewards ABC-style assessment before stabilization

Optimal sequence:

```text
CHECK_SCENE_SAFETY -> CHECK_BREATHING -> CONTROL_AIRWAY -> CALL_EMERGENCY -> CHECK_PULSE -> APPLY_PRESSURE -> MONITOR_PATIENT
```

### `anaphylaxis_medium`

- airway-driven allergic emergency
- requires breathing check before airway intervention

Optimal sequence:

```text
CHECK_SCENE_SAFETY -> CALL_EMERGENCY -> CHECK_BREATHING -> CONTROL_AIRWAY -> CHECK_PULSE -> MONITOR_PATIENT
```

### `choking_easy`

- obvious airway obstruction
- delay is heavily penalized

Optimal sequence:

```text
CHECK_RESPONSIVENESS -> CALL_EMERGENCY -> CONTROL_AIRWAY -> CHECK_BREATHING -> MONITOR_PATIENT
```

## Frontend

The React frontend now:

- groups actions into priority and secondary buckets
- uses the live backend `available_actions`
- highlights the next recommended action from the task flow
- marks actions already used in the current episode
- displays `reward_signal.reason`, `action_reward`, `time_decay_factor`, and `total`

Frontend source:

- `frontend/src/App.jsx`
- `frontend/src/styles.css`

## Training and Inference

### RL baseline

Train:

```bash
python train_rl.py
```

### LLM-first inference

Run:

```bash
python inference.py
```

The inference agent:

- calls the OpenAI-compatible client on every step
- asks for structured JSON with `reasoning` and `action`
- falls back to the RL table if the LLM output is invalid
- falls back again to a deterministic baseline if needed

Environment variables:

```bash
API_BASE_URL
MODEL_NAME
HF_TOKEN
OPENAI_API_KEY
```

## Local Setup

Backend:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Docker

```bash
docker build -t emergency-first-response-engine .
docker run --rm -p 7860:7860 emergency-first-response-engine
```

## Validation

Current validation script:

```bash
python validate_submission.py
```

## Current Limitations

The project is stronger than the initial version, but a few limits still matter:

- the RL baseline is tuned for earlier task dynamics and may need retraining or reward retuning for the new deceptive hard task
- scenario-aware `available_actions` improve realism, but they are still rule-based rather than fully physiology-driven
- the frontend depends on the backend’s current action ordering and is not yet split into reusable components
- some medically meaningful interventions are abstracted into coarse actions rather than detailed sub-skills
- the reward model is clinically guided but still simplified for deterministic benchmark use

## Recommended Next Enhancements

- retrain and retune the RL baseline against the deceptive hard task
- add episode traces/history export for debugging policies
- expose task-specific action hints separately from `available_actions`
- add frontend badges for invalid-action penalties and deterioration events
- extend trauma state with explicit shock markers such as skin signs or mental status changes over time

## Files Most Relevant To Change

- `environment/env.py`
- `environment/grader.py`
- `environment/tasks.py`
- `environment/models.py`
- `frontend/src/App.jsx`
- `frontend/src/styles.css`
- `app.py`
- `inference.py`
