---
title: Emergency First-Response Decision Engine
sdk: docker
app_port: 7860
tags:
  - openenv
  - fastapi
  - reinforcement-learning
  - decision-support
  - medical-ai
  - emergency-response
  - benchmark
  - agent-evaluation
  - pytorch
  - rl-environment
---

# Emergency First-Response Decision Engine

OpenEnv-compatible reinforcement learning environment for emergency decision support under uncertainty, time pressure, and clinical risk.

## Try It Instantly

**[https://huggingface.co/spaces/Venkat-023/OpenEnv](https://huggingface.co/spaces/Venkat-023/OpenEnv)**

No setup needed — open the link and start testing the environment directly from your browser.

---

## Problem Statement

In the first few minutes of a medical emergency, the difference between a good decision and a delayed or unsafe decision can determine whether a patient survives. In the real world, early responders are often not paramedics. They are bystanders, family members, security staff, transport workers, school staff, restaurant managers, or ordinary citizens who must act before professional medical help arrives.

Today, most agent benchmarks focus on web navigation, code generation, browsing, or abstract planning tasks. Very few environments test whether an AI system can reason through a time-critical real-world emergency with incomplete information, misleading surface cues, evolving patient state, and a strict requirement for safe action ordering.

This project addresses that gap.

The **Emergency First-Response Decision Engine** is designed as a realistic evaluation and training environment for AI agents that must support early emergency response. It simulates what humans actually do in urgent situations:

- assess the scene
- identify the most immediate life threat
- gather missing information
- choose the next best intervention
- avoid harmful or premature actions
- adapt as the patient deteriorates or stabilizes

This is not a game, puzzle, or toy simulator. It is a structured benchmark for testing whether an AI agent can behave like a reliable emergency decision-support assistant.

## Our Intuition

We built this project around a simple but powerful belief:

**If AI is going to be trusted in the real world, it must prove it can reason safely under pressure, not just answer questions in calm settings.**

Emergency care is one of the clearest examples of this challenge. A strong agent cannot rely on pattern matching alone. It must:

- reason sequentially
- handle partial observability
- distinguish between similar-looking but clinically different situations
- prioritize interventions by urgency
- accept that delay itself changes the state of the world

That is why this benchmark is built around deterioration, hidden state, state-aware action availability, clinically meaningful reward shaping, and deterministic task grading.

## End Goal

Our immediate goal is to build a **serious OpenEnv benchmark for first-response decision support** that can be used to evaluate and train agents on realistic emergency workflows.

Our long-term goal is larger:

We want to help create a future where AI systems can become dependable, auditable, and safe decision-support partners in high-stakes real-world settings. That does not mean replacing clinicians or emergency professionals. It means building agentic systems that can:

- support untrained responders during the first minutes of crisis
- provide structured guidance under uncertainty
- surface the right next step instead of generic advice
- be benchmarked transparently with reproducible scoring
- expose failure modes before deployment into sensitive domains

In other words, we want this project to contribute to a future where agent evaluation moves beyond convenience tasks and into domains that genuinely matter for human safety.

## What Makes This Different

This environment is designed to be memorable to judges and useful to researchers because it combines five properties that are rarely present together:

1. **Real-world utility** — It models a task humans actually perform during emergencies.
2. **Partial observability** — The full patient state is not visible at reset. Information must be earned by assessment.
3. **Clinically meaningful sequencing** — Actions are not just right or wrong; they can be correct but late, technically available but unsafe, or superficially sensible while missing the true life threat.
4. **Deceptive hard-case design** — The hardest trauma task intentionally hides the real danger behind misleading early cues.
5. **Deterministic evaluation** — The benchmark is reproducible, auditable, and suitable for consistent grading.

---

## Tasks (5 Total)

All five tasks are defined in `environment/tasks.py` and registered in `openenv.yaml`.

| Task ID | Difficulty | Scenario | Key Challenge |
|---|---|---|---|
| `cardiac_arrest_easy` | Easy | Collapse in airport terminal with AED nearby | Follow clear CPR + AED protocol without delay |
| `severe_bleeding_medium` | Medium | Kitchen laceration with rapid blood loss | Prioritize scene safety, bleeding control, and shock monitoring |
| `road_accident_hard` | Hard | Roadside crash with minor visible wound | Detect hidden internal hemorrhagic shock behind deceptive surface cues |
| `anaphylaxis_medium` | Medium | Severe allergic reaction with airway compromise | Assess breathing before airway intervention; activate help early |
| `choking_easy` | Easy | Elderly man choking in restaurant | Act fast — delays are heavily penalized in this task |

### 1. Cardiac Arrest — Easy

A recognizable public collapse scenario with AED access nearby. The agent must escalate quickly, confirm breathing status, start CPR, use AED, and monitor after intervention. Tests whether the agent can follow an obvious life-saving protocol without unnecessary delay.

### 2. Severe Bleeding — Medium

Visible hemorrhage with environmental hazards and circulation reassessment. The agent must account for hazards, escalate appropriately, control major bleeding, assess pulse, and monitor for shock. Tests multi-step trauma sequencing.

### 3. Road Accident with Hidden Shock — Hard

The most distinctive task in the benchmark. Minor visible bleeding masks evolving internal hemorrhagic shock plus airway compromise. The agent must avoid being fooled by surface appearance, work through ABC-style assessment, detect circulation compromise, and intervene before collapse. This is genuine sequential reasoning under misleading information.

### 4. Anaphylaxis — Medium

A severe allergic reaction with airway compromise. The agent must recognize the airway risk early, activate emergency services, and ensure breathing assessment precedes any airway intervention. Delayed airway support leads to progressive obstruction and cardiovascular collapse.

### 5. Choking — Easy

An obvious upper-airway obstruction where delay is especially dangerous. The agent must quickly establish responsiveness, call for help, and clear the airway. Every WAIT action is penalized. Inaction causes unconsciousness and arrest.

---

## Scoring

Scores are deterministic and bounded strictly inside (0, 1) for every task. The grader penalizes harmful actions, incorrect sequencing, and inefficiency.

| Factor | Weight |
|---|---|
| Correct action order (positional match) | 60% |
| Critical action coverage (unique hits) | 20% |
| Efficiency (steps used vs optimal) | 15% |
| Safety (no harmful or duplicate actions) | 5% |

Additional per-task penalties apply:

- `road_accident_hard` — extra penalty for skipping pulse check or starting CPR before assessment
- `anaphylaxis_medium` — extra penalty for airway intervention before breathing assessment
- `choking_easy` — cumulative penalty per WAIT action

---

## Environment Overview

The benchmark is implemented around the standard OpenEnv interface:

- `reset(task_id: str | None = None) -> Observation`
- `step(action: Action) -> tuple[Observation, float, bool, dict]`
- `state() -> InternalState`

Core files:

- `environment/env.py`
- `environment/models.py`
- `environment/tasks.py`
- `environment/grader.py`
- `openenv.yaml`

## OpenEnv Compliance

This repository implements the full typed environment interface expected by OpenEnv.

### Typed Models

Defined in `environment/models.py`:

- `Observation`
- `Action`
- `RewardSignal`
- `StepOutput`
- `InternalState`

### Core Methods

Defined in `environment/env.py`:

- `reset()`
- `step()`
- `state()`

### Metadata

Defined in `openenv.yaml`:

- environment name
- description
- entrypoint
- action model
- observation model
- reward model
- tasks and grader bindings

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

### Partial Observability

The environment intentionally hides parts of the patient state until the agent performs relevant checks.

Reveal rules:

- `CHECK_RESPONSIVENESS` reveals `conscious_status`
- `CHECK_BREATHING` reveals `breathing_status` and `airway_status`
- `CHECK_PULSE` reveals `pulse_status`
- `CHECK_SCENE_SAFETY` reveals `environment_context` and `risk_level`

This turns the benchmark from simple label prediction into sequential information-gathering under pressure.

## Action Space

The environment uses a discrete, structured action space:

- `CALL_EMERGENCY`
- `CHECK_SCENE_SAFETY`
- `CHECK_RESPONSIVENESS`
- `CHECK_BREATHING`
- `CHECK_PULSE`
- `START_CPR`
- `USE_AED`
- `APPLY_PRESSURE`
- `CONTROL_AIRWAY`
- `PLACE_RECOVERY_POSITION`
- `MONITOR_PATIENT`
- `WAIT`

These actions were chosen to reflect recognizable first-response decisions rather than arbitrary control tokens.

## Reward Design

The reward function provides dense signal across the full trajectory rather than only at episode termination.

Reward characteristics:

- positive reward for clinically appropriate actions
- positive reward for correct sequencing
- time-discounted reward for delayed critical interventions
- negative reward for unsafe actions
- negative reward for repeated actions without reassessment
- negative reward for deterioration caused by delay
- terminal bonus for stabilization
- terminal penalty for critical failure

Each step returns a structured `reward_signal` with:

- `action_reward`
- `reason`
- `time_decay_factor`
- `total`
- compatibility fields such as `value`, `components`, and `rationale`

This makes the reward not only useful for learning, but also inspectable for reviewers.

## Graders

Graders are deterministic and implemented in `environment/grader.py`.

Properties:

- task-specific
- programmatic
- reproducible
- bounded strictly inside `(0, 1)` for all trajectories
- sensitive to harmful actions, ordering mistakes, and inefficiency

## State-Aware Action Availability

The environment does not expose every action at every step.

Instead, `available_actions` are scenario- and state-aware. This improves:

- realism
- training signal quality
- UI clarity
- invalid action handling

Examples:

- `USE_AED` is only available after CPR has started in cardiac arrest
- airway actions become available after relevant assessment in airway-driven tasks
- the hard trauma task unlocks deeper interventions only after the agent gathers meaningful evidence

## Baseline Agents

The project includes two baseline styles:

### RL Baseline

Implemented in `rl_agent.py`:

- tabular Q-learning
- deterministic state encoding
- reproducible seed
- persistent Q-table artifact

Training:

```bash
python train_rl.py
```

### LLM-Driven Inference

Implemented in `inference.py`:

- uses the OpenAI Python client
- reads credentials from environment variables
- emits strict validator-compatible stdout logs
- can fall back safely if model output is invalid

Environment variables:

- `HF_TOKEN`
- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- `LOCAL_IMAGE_NAME`

For official hackathon submission, `HF_TOKEN` is treated as the mandatory primary credential for `inference.py`, while `API_BASE_URL` and `MODEL_NAME` use default values if not explicitly set.

## Inference Logging Contract

The inference script is designed to comply with strict evaluation parsers.

It emits only:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... rewards=...`

No extra stdout noise should be emitted during evaluation.

## API Surface

The FastAPI service in `app.py` exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve the frontend |
| `GET` | `/healthz` | Health check |
| `GET` | `/state` | Current internal environment state |
| `GET` | `/tasks` | Task metadata |
| `POST` | `/reset` | Start or reset an episode |
| `POST` | `/step` | Advance one action |
| `GET` | `/docs` | Swagger UI |

## Frontend

The React frontend provides a lightweight judge-friendly interface for:

- selecting a task
- resetting an episode
- choosing valid actions
- inspecting observation updates
- viewing reward explanations
- testing the environment quickly from one page

Frontend files:

- `frontend/src/App.jsx`
- `frontend/src/styles.css`

---

## Local Setup

Quickstart with Docker (recommended):

```bash
docker build -t emergency-first-response-engine .
docker run --rm -p 7860:7860 emergency-first-response-engine
# Open http://localhost:7860
```

Manual setup:

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

## Hugging Face Spaces

This repository is prepared for Docker-based Hugging Face Space deployment:

- root service responds on `${PORT:-7860}`
- frontend and backend are served from one container
- README includes Space metadata
- project is tagged with `openenv`

## Validation

Pre-submission validation:

```bash
python validate_submission.py
```

Recommended external checks:

- `openenv validate`
- `docker build`
- Space ping against `/reset`

---

## Limitations

- The environment models non-expert first-response, not advanced clinical care. Actions like medication administration are intentionally out of scope.
- Partial observability is rule-based, not stochastic. Noise models are not yet implemented.
- RL baseline uses tabular Q-learning; deep RL methods have not been evaluated yet.
- Tasks cover five scenarios; broader emergency coverage is planned in future work.

## Future Work

- Richer clinical cues and temporal trend summaries
- Broader first-aid and field-response scenarios such as burns, drowning, and seizure
- Deep RL baselines and policy-learning studies
- Human-evaluable episode traces
- Comparative evaluation across frontier LLMs and RL agents
- Safer, more interpretable decision-support benchmarking for real-world deployment research

---

## Why This Project Matters

This benchmark is ultimately about trust.

A powerful AI agent is not useful in the real world unless we can evaluate how it behaves when:

- information is incomplete
- time matters
- actions have safety consequences
- the obvious answer is not always the correct one

We believe future agent benchmarks must move beyond convenience tasks and begin measuring whether systems can act responsibly in domains where human welfare is directly at stake.

The Emergency First-Response Decision Engine is our contribution toward that future.

---

## Team

- **Venkat** — [GitHub: Venkat-023](https://github.com/Venkat-023)