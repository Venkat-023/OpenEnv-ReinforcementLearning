from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.models import Action, ActionType, Observation
from environment.tasks import TASKS
from rl_agent import DEFAULT_Q_TABLE_PATH, QLearningEmergencyAgent


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "emergency_first_response_decision_engine")


class BaselineEmergencyAgent:
    def __init__(self, client: OpenAI, model_name: str, rl_agent: QLearningEmergencyAgent | None = None) -> None:
        self._client = client
        self._model_name = model_name
        self._rl_agent = rl_agent

    def choose_action(self, observation: Observation) -> ActionType:
        if self._rl_agent is not None:
            return self._rl_agent.choose_action(observation, greedy=True)

        prompt = self._build_prompt(observation)
        try:
            completion = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0,
                max_tokens=16,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a first-response decision policy. Return exactly one action token from "
                            "the allowed action list and nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            candidate = (completion.choices[0].message.content or "").strip().split()[0]
            return ActionType(candidate)
        except Exception:
            return self._fallback_policy(observation)

    def _build_prompt(self, observation: Observation) -> str:
        return (
            f"task_id={observation.task_id}\n"
            f"time_elapsed={observation.time_elapsed}\n"
            f"risk_level={observation.risk_level}\n"
            f"conscious_status={observation.patient_condition.conscious_status.value}\n"
            f"breathing_status={observation.patient_condition.breathing_status.value}\n"
            f"bleeding_severity={observation.patient_condition.bleeding_severity.value}\n"
            f"pulse_status={observation.patient_condition.pulse_status.value}\n"
            f"airway_status={observation.patient_condition.airway_status.value}\n"
            f"actions_taken={[action.value for action in observation.actions_taken]}\n"
            f"available_actions={[action.value for action in observation.available_actions]}\n"
            "Respond with one action token."
        )

    def _fallback_policy(self, observation: Observation) -> ActionType:
        planned_actions = TASKS[observation.task_id].optimal_sequence
        next_index = len(observation.actions_taken)
        if next_index < len(planned_actions):
            return planned_actions[next_index]
        return ActionType.MONITOR_PATIENT


def build_client() -> OpenAI:
    api_key = OPENAI_API_KEY or HF_TOKEN or "missing-token"
    return OpenAI(base_url=API_BASE_URL.rstrip("/"), api_key=api_key)


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    client = build_client()
    learned_agent = QLearningEmergencyAgent()
    has_learned_policy = learned_agent.load(DEFAULT_Q_TABLE_PATH)
    agent = BaselineEmergencyAgent(
        client=client,
        model_name=MODEL_NAME,
        rl_agent=learned_agent if has_learned_policy else None,
    )
    environment = EmergencyFirstResponseDecisionEngine()

    for task_id in TASKS:
        rewards: list[float] = []
        steps_taken = 0
        success = False
        log_start(task_id)

        try:
            observation = environment.reset(task_id)
            done = False

            while not done:
                step_number = steps_taken + 1
                action = agent.choose_action(observation)
                error: str | None = None

                try:
                    observation, reward, done, info = environment.step(Action(action_type=action))
                    success = bool(info.get("success", False))
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = str(exc)

                rewards.append(reward)
                steps_taken = step_number
                log_step(step_number, action.value, reward, done, error)

        finally:
            log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    main()
