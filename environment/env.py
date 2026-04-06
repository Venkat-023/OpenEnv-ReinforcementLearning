from __future__ import annotations

from copy import deepcopy
from typing import Any

from environment.grader import EmergencyTaskGrader
from environment.models import (
    Action,
    ActionType,
    AirwayStatus,
    BleedingSeverity,
    BreathingStatus,
    ConsciousStatus,
    InternalState,
    Observation,
    PatientCondition,
    PulseStatus,
    RewardSignal,
)
from environment.tasks import DEFAULT_TASK_ID, TASKS


CRITICAL_ACTIONS: set[ActionType] = {
    ActionType.CALL_EMERGENCY,
    ActionType.START_CPR,
    ActionType.APPLY_PRESSURE,
    ActionType.CONTROL_AIRWAY,
    ActionType.USE_AED,
}

REVEALS: dict[ActionType, set[str]] = {
    ActionType.CHECK_RESPONSIVENESS: {"conscious_status"},
    ActionType.CHECK_BREATHING: {"breathing_status", "airway_status"},
    ActionType.CHECK_PULSE: {"pulse_status"},
    ActionType.CHECK_SCENE_SAFETY: {"environment_context", "risk_level"},
}

VISIBLE_FROM_START: dict[str, set[str]] = {
    "severe_bleeding_medium": {"bleeding_severity"},
}


class EmergencyFirstResponseDecisionEngine:
    def __init__(self, default_task_id: str = DEFAULT_TASK_ID) -> None:
        self._grader = EmergencyTaskGrader()
        self._default_task_id = default_task_id
        self._state = self._build_state(default_task_id)

    def reset(self, task_id: str | None = None) -> Observation:
        selected_task = task_id or self._default_task_id
        if selected_task not in TASKS:
            raise ValueError(f"Unknown task_id: {selected_task}")
        self._state = self._build_state(selected_task)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._state.done:
            observation = self._build_observation()
            return observation, 0.0, True, self._build_info(termination_locked=True)

        available_actions = self._available_actions()
        if action.action_type not in available_actions:
            self._state.last_action_effect = "Invalid action ignored."
            observation = self._build_observation()
            reward_signal = self._build_reward_signal(
                action_reward=-0.2,
                total_reward=-0.2,
                time_decay_factor=1.0,
                reason=self._grader.explain_reward(
                    self._state.task_id,
                    action.action_type,
                    self._state.model_copy(deep=True),
                    self._state.model_copy(deep=True),
                    -0.2,
                    1.0,
                    invalid_action=True,
                ),
                components={"invalid_action_penalty": -0.2},
            )
            info = self._build_info(reward_signal=reward_signal)
            info["error"] = "invalid_action"
            return observation, -0.2, False, info

        prior_state = self._state.model_copy(deep=True)
        reward = -0.2
        reward_components: dict[str, float] = {"base_step_cost": -0.2}
        effect_messages: list[str] = []
        chosen_action = action.action_type
        prior_actions = list(self._state.actions_taken)
        time_decay_factor = 1.0

        if prior_actions and prior_actions[-1] == chosen_action:
            reward -= 0.6
            reward_components["repeat_penalty"] = -0.6
            effect_messages.append("Action repeated without reassessment.")

        action_delta, action_metadata = self._apply_action(chosen_action, effect_messages)
        reward += action_delta
        reward_components["action_reward"] = action_delta
        reward_components.update(action_metadata)
        time_decay_factor = action_metadata.get("critical_action_time_factor", 1.0)

        self._state.actions_taken.append(chosen_action)
        self._state.revealed_fields.update(REVEALS.get(chosen_action, set()))
        self._update_critical_action_counters(chosen_action)

        deterioration_penalty = self._apply_deterioration(effect_messages)
        if deterioration_penalty != 0.0:
            reward += deterioration_penalty
            reward_components["deterioration_penalty"] = deterioration_penalty

        self._state.time_elapsed += 1

        progression_delta = self._apply_progression(effect_messages)
        reward += progression_delta
        reward_components["progression_delta"] = progression_delta

        self._update_observed_condition()
        self._check_termination()

        if self._state.done:
            terminal_delta = 5.0 if self._state.success else -5.0
            reward += terminal_delta
            reward_components["terminal_delta"] = terminal_delta

        self._state.last_action_effect = " ".join(effect_messages) if effect_messages else "No meaningful change."
        reason = self._grader.explain_reward(
            self._state.task_id,
            chosen_action,
            prior_state,
            self._state.model_copy(deep=True),
            action_delta,
            time_decay_factor,
        )
        observation = self._build_observation()
        reward_signal = self._build_reward_signal(
            action_reward=round(action_delta, 3),
            total_reward=round(reward, 3),
            time_decay_factor=time_decay_factor,
            reason=reason,
            components={key: round(value, 3) for key, value in reward_components.items()},
        )
        info = self._build_info(reward_signal=reward_signal)
        return observation, round(reward, 3), self._state.done, info

    def state(self) -> InternalState:
        return self._state.model_copy(deep=True)

    def _build_state(self, task_id: str) -> InternalState:
        task = TASKS[task_id]
        state = InternalState(
            task_id=task.task_id,
            difficulty=task.difficulty,
            scenario_summary=task.scenario_summary,
            true_condition=task.initial_true_condition.model_copy(deep=True),
            observed_condition=task.initial_observed_condition.model_copy(deep=True),
            environment_context=task.environment_context.model_copy(deep=True),
            max_steps=task.max_steps,
            optimal_sequence=deepcopy(task.optimal_sequence),
            hidden_notes=deepcopy(task.hidden_notes),
            revealed_fields=set(VISIBLE_FROM_START.get(task_id, set())),
        )
        self._synchronize_observed_condition(state)
        return state

    def _build_observation(self) -> Observation:
        visible_condition = self._visible_patient_condition()
        environment_context: object
        if "environment_context" in self._state.revealed_fields:
            environment_context = self._state.environment_context.model_copy(deep=True)
        else:
            environment_context = "unknown"

        risk_level = self._risk_level() if "risk_level" in self._state.revealed_fields else "unknown"
        return Observation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            scenario_summary=self._state.scenario_summary,
            patient_condition=visible_condition,
            time_elapsed=self._state.time_elapsed,
            actions_taken=list(self._state.actions_taken),
            environment_context=environment_context,
            available_actions=self._available_actions(),
            last_action_effect=self._state.last_action_effect,
            risk_level=risk_level,
        )

    def _build_info(
        self,
        termination_locked: bool = False,
        reward_signal: RewardSignal | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty.value,
            "termination_reason": self._state.termination_reason,
            "success": self._state.success,
            "score_so_far": self._grader.grade_task(self._state.task_id, self._state.actions_taken),
            "optimal_sequence": [action.value for action in self._state.optimal_sequence],
            "termination_locked": termination_locked,
            "reward_signal": reward_signal.model_dump() if reward_signal else None,
        }

    def _build_reward_signal(
        self,
        action_reward: float,
        total_reward: float,
        time_decay_factor: float,
        reason: str,
        components: dict[str, float],
    ) -> RewardSignal:
        return RewardSignal(
            action_reward=round(action_reward, 3),
            reason=reason,
            time_decay_factor=round(time_decay_factor, 3),
            total=round(total_reward, 3),
            value=round(total_reward, 3),
            components=components,
            rationale=reason,
        )

    def _apply_action(self, action: ActionType, effects: list[str]) -> tuple[float, dict[str, float]]:
        task_id = self._state.task_id
        reward = 0.0
        metadata: dict[str, float] = {}

        if action == ActionType.CHECK_SCENE_SAFETY:
            if self._state.environment_context.hazards:
                effects.append("Hazards identified and managed as well as possible.")
                reward += 1.2
            else:
                effects.append("Scene appears safe.")
                reward += 0.4
            if task_id == "road_accident_hard":
                reward += 0.6
        elif action == ActionType.CALL_EMERGENCY:
            if not self._state.emergency_called:
                self._state.emergency_called = True
                effects.append("Emergency services alerted.")
                reward += 2.0
                if task_id == "road_accident_hard" and self._state.airway_controlled:
                    reward += 0.4
            else:
                effects.append("Emergency services were already contacted.")
                reward -= 0.3
        elif action == ActionType.CHECK_RESPONSIVENESS:
            self._state.responsiveness_checked = True
            effects.append("Responsiveness assessed.")
            reward += 0.7
        elif action == ActionType.CHECK_BREATHING:
            self._state.breathing_checked = True
            effects.append("Breathing assessed.")
            reward += 1.0
            if task_id == "road_accident_hard":
                reward += 0.5
        elif action == ActionType.CHECK_PULSE:
            self._state.pulse_checked = True
            effects.append("Pulse assessed.")
            reward += 0.8
            if task_id == "road_accident_hard":
                reward += 0.7
        elif action == ActionType.START_CPR:
            if task_id == "cardiac_arrest_easy":
                if self._state.true_condition.breathing_status == BreathingStatus.ABSENT:
                    self._state.cpr_started = True
                    effects.append("CPR started promptly.")
                    reward += 2.5
                else:
                    effects.append("CPR is not indicated while the patient is breathing.")
                    reward -= 1.6
            else:
                if (
                    self._state.true_condition.breathing_status == BreathingStatus.ABSENT
                    and self._state.true_condition.pulse_status == PulseStatus.ABSENT
                ):
                    self._state.cpr_started = True
                    effects.append("CPR started after recognizing cardiac arrest.")
                    reward += 2.0
                else:
                    effects.append("CPR would be unsafe in the current state.")
                    reward -= 1.8
        elif action == ActionType.USE_AED:
            if task_id == "cardiac_arrest_easy" and self._state.cpr_started:
                self._state.aed_used = True
                self._state.true_condition.pulse_status = PulseStatus.WEAK
                effects.append("AED applied after CPR began.")
                reward += 2.3
            else:
                effects.append("AED use is ineffective or premature here.")
                reward -= 1.4
        elif action == ActionType.APPLY_PRESSURE:
            if (
                task_id == "road_accident_hard"
                and (not self._state.pulse_checked or not self._state.breathing_checked)
            ):
                effects.append("Superficial bleeding control delays recognition of hidden shock.")
                reward -= 1.4
            elif self._state.true_condition.bleeding_severity in {BleedingSeverity.SEVERE, BleedingSeverity.CRITICAL}:
                self._state.pressure_applied = True
                self._state.true_condition.bleeding_severity = BleedingSeverity.MODERATE
                effects.append("Direct pressure reduces bleeding.")
                reward += 2.2
            elif self._state.true_condition.bleeding_severity == BleedingSeverity.MODERATE:
                self._state.pressure_applied = True
                self._state.true_condition.bleeding_severity = BleedingSeverity.MILD
                effects.append("Bleeding further reduced.")
                reward += 1.0
            else:
                effects.append("No major external bleeding to compress.")
                reward -= 0.8
        elif action == ActionType.CONTROL_AIRWAY:
            if self._state.true_condition.airway_status == AirwayStatus.COMPROMISED:
                self._state.airway_controlled = True
                self._state.true_condition.airway_status = AirwayStatus.CLEAR
                if self._state.true_condition.breathing_status == BreathingStatus.ABSENT:
                    self._state.true_condition.breathing_status = BreathingStatus.SHALLOW
                elif self._state.true_condition.breathing_status == BreathingStatus.SHALLOW:
                    self._state.true_condition.breathing_status = BreathingStatus.NORMAL
                effects.append("Airway supported and breathing improves.")
                reward += 2.0
                if task_id == "road_accident_hard" and self._state.breathing_checked:
                    reward += 0.8
            else:
                effects.append("Airway support provided, but airway was already clear.")
                reward -= 0.2
        elif action == ActionType.PLACE_RECOVERY_POSITION:
            if (
                self._state.true_condition.breathing_status != BreathingStatus.ABSENT
                and self._state.true_condition.conscious_status == ConsciousStatus.UNCONSCIOUS
                and self._state.task_id not in {"road_accident_hard", "choking_easy", "anaphylaxis_medium"}
            ):
                effects.append("Recovery position protects the airway.")
                reward += 1.0
            else:
                effects.append("Recovery position is unsafe or not indicated.")
                reward -= 1.5
        elif action == ActionType.MONITOR_PATIENT:
            effects.append("Patient reassessed and trends monitored.")
            reward += 0.9 if self._state.emergency_called else 0.2
        elif action == ActionType.WAIT:
            effects.append("No intervention performed while the condition evolves.")
            reward -= 1.2

        if action in CRITICAL_ACTIONS:
            if self._state.steps_to_first_critical_action is None:
                self._state.steps_to_first_critical_action = self._state.time_elapsed
            if reward > 0:
                time_factor = max(0.5, 1.0 - 0.08 * self._state.time_elapsed)
                metadata["critical_action_raw_delta"] = reward
                metadata["critical_action_time_factor"] = time_factor
                metadata["critical_action_delay_steps"] = float(self._state.time_elapsed)
                reward *= time_factor

        return reward, metadata

    def _apply_progression(self, effects: list[str]) -> float:
        reward = 0.0
        task_id = self._state.task_id

        if task_id == "cardiac_arrest_easy":
            if not self._state.cpr_started:
                reward -= 1.8
                effects.append("Ongoing cardiac arrest without CPR worsens survival odds.")
            if self._state.cpr_started and not self._state.aed_used:
                reward += 0.5
                effects.append("CPR is maintaining some circulation while waiting for defibrillation.")
            if self._state.aed_used:
                self._state.true_condition.breathing_status = BreathingStatus.SHALLOW
                self._state.true_condition.pulse_status = PulseStatus.WEAK
                effects.append("After defibrillation, the patient shows signs of circulation.")
        elif task_id == "severe_bleeding_medium":
            if not self._state.pressure_applied:
                reward -= 1.4
                if self._state.time_elapsed >= 2:
                    self._state.true_condition.bleeding_severity = BleedingSeverity.CRITICAL
                    self._state.true_condition.pulse_status = PulseStatus.WEAK
                    effects.append("Bleeding remains uncontrolled and worsens.")
                if self._state.time_elapsed >= 4:
                    self._state.true_condition.conscious_status = ConsciousStatus.CONFUSED
                    effects.append("Signs of shock are developing.")
            else:
                if self._state.true_condition.bleeding_severity == BleedingSeverity.MODERATE:
                    self._state.true_condition.pulse_status = PulseStatus.NORMAL
                reward += 0.5
                effects.append("Bleeding control is preserving circulation.")
        elif task_id == "road_accident_hard":
            if not self._state.airway_controlled:
                reward -= 0.8
                if self._state.time_elapsed >= 3:
                    self._state.true_condition.breathing_status = BreathingStatus.ABSENT
                    effects.append("Unmanaged airway compromise worsens ventilation.")
            else:
                reward += 0.6
                effects.append("Airway control improves oxygenation.")

            if not self._state.pulse_checked:
                reward -= 0.9
                if self._state.time_elapsed >= 2:
                    self._state.true_condition.conscious_status = ConsciousStatus.UNCONSCIOUS
                    self._state.true_condition.pulse_status = PulseStatus.WEAK
                    effects.append("Occult hemorrhagic shock is missed while circulation goes unassessed.")
            else:
                reward += 0.5
                effects.append("Circulation assessment exposes signs of hidden shock.")

            if not self._state.pressure_applied and self._state.time_elapsed >= 6:
                self._state.true_condition.pulse_status = PulseStatus.ABSENT
                reward -= 1.8
                effects.append("Shock progresses to circulatory collapse without stabilization.")
            elif self._state.pressure_applied:
                self._state.true_condition.bleeding_severity = BleedingSeverity.MODERATE
                self._state.true_condition.pulse_status = PulseStatus.NORMAL
                reward += 0.6
                effects.append("Stabilization slows internal hemorrhagic decline.")
        elif task_id == "anaphylaxis_medium":
            if not self._state.airway_controlled:
                reward -= 0.9
                if self._state.time_elapsed >= 3:
                    self._state.true_condition.breathing_status = BreathingStatus.ABSENT
                    self._state.true_condition.conscious_status = ConsciousStatus.UNCONSCIOUS
                    effects.append("Airway swelling worsens and breathing becomes ineffective.")
                else:
                    self._state.true_condition.breathing_status = BreathingStatus.SHALLOW
            else:
                self._state.true_condition.breathing_status = BreathingStatus.NORMAL
                reward += 0.6
                effects.append("Airway support improves ventilation while awaiting advanced care.")

            if not self._state.emergency_called:
                reward -= 0.4
            if self._state.time_elapsed >= 5 and self._state.true_condition.breathing_status == BreathingStatus.ABSENT:
                self._state.true_condition.pulse_status = PulseStatus.ABSENT
                reward -= 1.8
                effects.append("Untreated respiratory failure progresses toward circulatory collapse.")
        elif task_id == "choking_easy":
            if not self._state.airway_controlled:
                reward -= 1.1
                self._state.true_condition.airway_status = AirwayStatus.COMPROMISED
                self._state.true_condition.breathing_status = BreathingStatus.ABSENT
                if self._state.time_elapsed >= 3:
                    self._state.true_condition.conscious_status = ConsciousStatus.UNCONSCIOUS
                    effects.append("Persistent airway obstruction causes loss of consciousness.")
                if self._state.time_elapsed >= 4:
                    self._state.true_condition.pulse_status = PulseStatus.ABSENT
                    reward -= 2.0
                    effects.append("Complete obstruction progresses to cardiac arrest.")
            else:
                self._state.true_condition.breathing_status = BreathingStatus.NORMAL
                self._state.true_condition.pulse_status = PulseStatus.NORMAL
                reward += 0.6
                effects.append("Relieved obstruction restores air movement.")

        return reward

    def _apply_deterioration(self, effects: list[str]) -> float:
        if self._state.steps_without_critical_action < 3:
            return 0.0

        penalty = 0.0
        task_id = self._state.task_id

        if task_id in {"cardiac_arrest_easy", "road_accident_hard"}:
            new_breathing = self._degrade_breathing(self._state.true_condition.breathing_status)
            if new_breathing != self._state.true_condition.breathing_status:
                self._state.true_condition.breathing_status = new_breathing
                penalty -= 0.15
                effects.append("Delay in critical care worsens breathing status.")

        if task_id in {"severe_bleeding_medium", "road_accident_hard"}:
            new_bleeding = self._worsen_bleeding(self._state.true_condition.bleeding_severity)
            if new_bleeding != self._state.true_condition.bleeding_severity:
                self._state.true_condition.bleeding_severity = new_bleeding
                penalty -= 0.15
                effects.append("Delay in critical care allows bleeding to worsen.")

        return penalty

    def _update_observed_condition(self) -> None:
        self._synchronize_observed_condition(self._state)

    def _synchronize_observed_condition(self, state: InternalState) -> None:
        true_condition = state.true_condition
        revealed = state.revealed_fields
        state.observed_condition = PatientCondition(
            conscious_status=(
                true_condition.conscious_status
                if "conscious_status" in revealed
                else ConsciousStatus.UNKNOWN
            ),
            breathing_status=(
                true_condition.breathing_status
                if "breathing_status" in revealed
                else BreathingStatus.UNKNOWN
            ),
            bleeding_severity=(
                true_condition.bleeding_severity
                if "bleeding_severity" in revealed
                else BleedingSeverity.UNKNOWN
            ),
            pulse_status=true_condition.pulse_status if "pulse_status" in revealed else PulseStatus.UNKNOWN,
            airway_status=true_condition.airway_status if "airway_status" in revealed else AirwayStatus.UNKNOWN,
        )

    def _check_termination(self) -> None:
        if self._state.time_elapsed >= self._state.max_steps:
            self._state.done = True
            self._state.termination_reason = "max_steps_reached"
            self._state.success = False
            return

        if self._state.task_id == "cardiac_arrest_easy":
            if (
                self._state.emergency_called
                and self._state.cpr_started
                and self._state.aed_used
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.time_elapsed >= 4 and not self._state.cpr_started:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "severe_bleeding_medium":
            if (
                self._state.emergency_called
                and self._state.pressure_applied
                and ActionType.CHECK_PULSE in self._state.actions_taken
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
                and self._state.time_elapsed >= 5
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif (
                self._state.true_condition.bleeding_severity == BleedingSeverity.CRITICAL
                and self._state.time_elapsed >= 5
                and not self._state.pressure_applied
            ):
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "road_accident_hard":
            if (
                ActionType.CHECK_SCENE_SAFETY in self._state.actions_taken
                and self._state.breathing_checked
                and self._state.airway_controlled
                and self._state.emergency_called
                and self._state.pulse_checked
                and self._state.pressure_applied
                and self._state.actions_taken
                and self._state.actions_taken[-1] == ActionType.MONITOR_PATIENT
                and self._state.time_elapsed >= 7
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.true_condition.pulse_status == PulseStatus.ABSENT:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "anaphylaxis_medium":
            if (
                self._state.emergency_called
                and self._state.breathing_checked
                and self._state.airway_controlled
                and ActionType.CHECK_PULSE in self._state.actions_taken
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
                and self._state.time_elapsed >= 6
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.true_condition.pulse_status == PulseStatus.ABSENT:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False
        elif self._state.task_id == "choking_easy":
            if (
                self._state.responsiveness_checked
                and self._state.emergency_called
                and self._state.airway_controlled
                and self._state.breathing_checked
                and ActionType.MONITOR_PATIENT in self._state.actions_taken
                and self._state.time_elapsed >= 5
            ):
                self._state.done = True
                self._state.termination_reason = "patient_stabilized"
                self._state.success = True
            elif self._state.true_condition.pulse_status == PulseStatus.ABSENT:
                self._state.done = True
                self._state.termination_reason = "critical_worsening"
                self._state.success = False

    def _risk_level(self) -> str:
        condition = self._state.true_condition
        if (
            condition.pulse_status == PulseStatus.ABSENT
            or condition.breathing_status == BreathingStatus.ABSENT
            or condition.bleeding_severity == BleedingSeverity.CRITICAL
            or condition.airway_status == AirwayStatus.COMPROMISED
        ):
            return "critical"
        if (
            condition.bleeding_severity == BleedingSeverity.SEVERE
            or condition.breathing_status == BreathingStatus.SHALLOW
            or condition.pulse_status == PulseStatus.WEAK
        ):
            return "high"
        return "moderate"

    def _available_actions(self) -> list[ActionType]:
        state = self._state
        actions: list[ActionType] = [ActionType.WAIT, ActionType.MONITOR_PATIENT]

        if not state.emergency_called:
            actions.append(ActionType.CALL_EMERGENCY)

        if state.task_id == "cardiac_arrest_easy":
            actions.extend(
                [
                    ActionType.CHECK_RESPONSIVENESS,
                    ActionType.CHECK_BREATHING,
                ]
            )
            if state.breathing_checked and not state.cpr_started:
                actions.append(ActionType.START_CPR)
            if state.cpr_started and not state.aed_used:
                actions.append(ActionType.USE_AED)

        elif state.task_id == "severe_bleeding_medium":
            actions.extend(
                [
                    ActionType.CHECK_SCENE_SAFETY,
                    ActionType.CHECK_RESPONSIVENESS,
                    ActionType.CHECK_PULSE,
                    ActionType.APPLY_PRESSURE,
                ]
            )
            if state.time_elapsed >= 2:
                actions.append(ActionType.CHECK_BREATHING)

        elif state.task_id == "road_accident_hard":
            if ActionType.CHECK_SCENE_SAFETY not in state.actions_taken:
                actions.append(ActionType.CHECK_SCENE_SAFETY)
            actions.append(ActionType.CHECK_BREATHING)
            if state.breathing_checked:
                actions.append(ActionType.CONTROL_AIRWAY)
                actions.append(ActionType.CHECK_PULSE)
            if state.pulse_checked:
                actions.append(ActionType.APPLY_PRESSURE)

        elif state.task_id == "anaphylaxis_medium":
            actions.extend(
                [
                    ActionType.CHECK_SCENE_SAFETY,
                    ActionType.CHECK_BREATHING,
                    ActionType.CHECK_PULSE,
                ]
            )
            if state.breathing_checked:
                actions.append(ActionType.CONTROL_AIRWAY)

        elif state.task_id == "choking_easy":
            actions.extend(
                [
                    ActionType.CHECK_RESPONSIVENESS,
                    ActionType.CHECK_BREATHING,
                ]
            )
            if state.responsiveness_checked or state.breathing_checked:
                actions.append(ActionType.CONTROL_AIRWAY)

        if (
            state.true_condition.conscious_status == ConsciousStatus.UNCONSCIOUS
            and state.true_condition.breathing_status != BreathingStatus.ABSENT
        ):
            actions.append(ActionType.PLACE_RECOVERY_POSITION)

        if state.task_id not in {"cardiac_arrest_easy", "anaphylaxis_medium", "choking_easy"}:
            actions.append(ActionType.CHECK_SCENE_SAFETY)

        ordered_unique: list[ActionType] = []
        for action in actions:
            if action not in ordered_unique:
                ordered_unique.append(action)
        return ordered_unique

    def _update_critical_action_counters(self, action: ActionType) -> None:
        if action in CRITICAL_ACTIONS:
            self._state.steps_without_critical_action = 0
        else:
            self._state.steps_without_critical_action += 1

    def _visible_patient_condition(self) -> PatientCondition:
        self._synchronize_observed_condition(self._state)
        return self._state.observed_condition.model_copy(deep=True)

    def _degrade_breathing(self, status: BreathingStatus) -> BreathingStatus:
        if status == BreathingStatus.NORMAL:
            return BreathingStatus.SHALLOW
        if status == BreathingStatus.SHALLOW:
            return BreathingStatus.ABSENT
        return status

    def _worsen_bleeding(self, severity: BleedingSeverity) -> BleedingSeverity:
        if severity == BleedingSeverity.MILD:
            return BleedingSeverity.MODERATE
        if severity == BleedingSeverity.MODERATE:
            return BleedingSeverity.SEVERE
        if severity == BleedingSeverity.SEVERE:
            return BleedingSeverity.CRITICAL
        return severity
