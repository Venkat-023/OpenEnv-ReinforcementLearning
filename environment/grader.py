from __future__ import annotations

from typing import Iterable

from environment.models import (
    ActionType,
    AirwayStatus,
    BleedingSeverity,
    BreathingStatus,
    InternalState,
    PulseStatus,
)
from environment.tasks import TASKS


HARMFUL_ACTIONS: dict[str, set[ActionType]] = {
    "cardiac_arrest_easy": {
        ActionType.WAIT,
        ActionType.PLACE_RECOVERY_POSITION,
        ActionType.APPLY_PRESSURE,
    },
    "severe_bleeding_medium": {
        ActionType.START_CPR,
        ActionType.USE_AED,
        ActionType.WAIT,
    },
    "road_accident_hard": {
        ActionType.WAIT,
        ActionType.PLACE_RECOVERY_POSITION,
        ActionType.USE_AED,
    },
    "anaphylaxis_medium": {
        ActionType.WAIT,
        ActionType.START_CPR,
        ActionType.USE_AED,
    },
    "choking_easy": {
        ActionType.WAIT,
        ActionType.APPLY_PRESSURE,
        ActionType.USE_AED,
    },
}


class EmergencyTaskGrader:
    def grade_task(self, task_id: str, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        if task_id == "cardiac_arrest_easy":
            return self.grade_easy(normalized)
        if task_id == "severe_bleeding_medium":
            return self.grade_medium(normalized)
        if task_id == "road_accident_hard":
            return self.grade_hard(normalized)
        if task_id == "anaphylaxis_medium":
            return self.grade_anaphylaxis(normalized)
        if task_id == "choking_easy":
            return self.grade_choking(normalized)
        raise KeyError(f"Unknown task_id: {task_id}")

    def explain_reward(
        self,
        task_id: str,
        action: ActionType,
        prior_state: InternalState,
        current_state: InternalState,
        action_reward: float,
        time_decay_factor: float,
        invalid_action: bool = False,
    ) -> str:
        if invalid_action:
            return "Submitted action is not currently available - the environment ignores it and applies a safety penalty"

        if action == ActionType.CHECK_SCENE_SAFETY:
            if prior_state.environment_context.hazards:
                return "Checked scene safety before patient contact - aligns with first-responder hazard control"
            return "Checked scene safety early - appropriate even when no immediate hazard is found"

        if action == ActionType.CALL_EMERGENCY:
            if prior_state.emergency_called:
                return "Repeated emergency activation adds no clinical value after help is already on the way"
            return "Called emergency services early - appropriate escalation for time-critical illness or trauma"

        if action == ActionType.CHECK_RESPONSIVENESS:
            return "Assessed responsiveness first - helps establish neurological status before intervention"

        if action == ActionType.CHECK_BREATHING:
            if task_id == "road_accident_hard":
                return "Checked breathing before committing to treatment - appropriate in a deceptive trauma presentation"
            return "Checked breathing before intervention - aligns with ABC protocol"

        if action == ActionType.CHECK_PULSE:
            if task_id == "road_accident_hard":
                return "Assessed circulation to look for occult shock - this is how hidden internal bleeding becomes apparent"
            return "Checked pulse to evaluate circulation - clinically appropriate reassessment step"

        if action == ActionType.START_CPR:
            if (
                prior_state.true_condition.pulse_status != PulseStatus.ABSENT
                or prior_state.true_condition.breathing_status != BreathingStatus.ABSENT
            ):
                return "Performed CPR without confirmed pulseless apnea - violates basic life support sequence"
            if not prior_state.breathing_checked and not prior_state.pulse_checked:
                return "Started CPR before assessment - correct only by luck, not by proper resuscitation sequence"
            return "Started CPR after identifying pulseless apnea - appropriate life-saving intervention"

        if action == ActionType.USE_AED:
            if task_id != "cardiac_arrest_easy" or not prior_state.cpr_started:
                return "USE_AED applied without confirmed cardiac arrest - contraindicated and potentially harmful"
            return "Applied AED after confirming arrest and starting CPR - correct defibrillation sequence"

        if action == ActionType.APPLY_PRESSURE:
            if prior_state.true_condition.bleeding_severity in {BleedingSeverity.SEVERE, BleedingSeverity.CRITICAL}:
                return "Applied direct pressure to major hemorrhage - correct first-line bleeding control"
            if task_id == "road_accident_hard":
                return "Focused on minor visible bleeding while the major threat is occult shock - this is too superficial"
            return "Applied pressure when there is no major external hemorrhage - limited clinical benefit"

        if action == ActionType.CONTROL_AIRWAY:
            if prior_state.true_condition.airway_status == AirwayStatus.COMPROMISED:
                return "Supported the airway after recognizing compromise - appropriate airway-first stabilization"
            return "Attempted airway intervention despite a clear airway - low-value action compared with reassessment"

        if action == ActionType.PLACE_RECOVERY_POSITION:
            return "Used recovery position outside the safest indication window - this can delay more urgent care"

        if action == ActionType.MONITOR_PATIENT:
            if prior_state.emergency_called:
                return "Monitored after initial stabilization - appropriate ongoing reassessment"
            return "Monitoring before definitive action delays time-critical care"

        if action == ActionType.WAIT:
            return "Delayed intervention during an unstable emergency - clinically unsafe because deterioration can progress"

        if action_reward < 0:
            return "Action does not fit the patient's current physiology and is penalized for poor clinical sequencing"

        if time_decay_factor < 1.0:
            return "Action is clinically appropriate, but delayed intervention reduced its benefit"

        return "Action is clinically appropriate for the current emergency state"

    def grade_easy(self, actions: Iterable[ActionType | str]) -> float:
        return self._base_grade("cardiac_arrest_easy", self._normalize(actions))

    def grade_medium(self, actions: Iterable[ActionType | str]) -> float:
        return self._base_grade("severe_bleeding_medium", self._normalize(actions))

    def grade_hard(self, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        score = self._base_grade("road_accident_hard", normalized)
        if ActionType.CHECK_PULSE not in normalized:
            score -= 0.18
        elif ActionType.CHECK_BREATHING in normalized and normalized.index(ActionType.CHECK_PULSE) < normalized.index(
            ActionType.CHECK_BREATHING
        ):
            score -= 0.08
        if normalized and normalized[0] == ActionType.START_CPR:
            score -= 0.15
        return round(max(0.0, min(1.0, score)), 4)

    def grade_anaphylaxis(self, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        score = self._base_grade("anaphylaxis_medium", normalized)
        if ActionType.CONTROL_AIRWAY in normalized:
            airway_index = normalized.index(ActionType.CONTROL_AIRWAY)
            if ActionType.CHECK_BREATHING not in normalized[:airway_index]:
                score -= 0.2
        return round(max(0.0, min(1.0, score)), 4)

    def grade_choking(self, actions: Iterable[ActionType | str]) -> float:
        normalized = self._normalize(actions)
        score = self._base_grade("choking_easy", normalized)
        wait_count = sum(1 for action in normalized if action == ActionType.WAIT)
        score -= 0.12 * wait_count
        return round(max(0.0, min(1.0, score)), 4)

    def _base_grade(self, task_id: str, actions: list[ActionType]) -> float:
        task = TASKS[task_id]
        optimal = task.optimal_sequence
        if not actions:
            return 0.0

        matched_positions = sum(
            1 for index, action in enumerate(actions[: len(optimal)]) if action == optimal[index]
        )
        unique_critical_hits = len(set(actions).intersection(set(optimal)))
        harmful_count = sum(1 for action in actions if action in HARMFUL_ACTIONS[task_id])
        duplicate_count = max(0, len(actions) - len(set(actions)))

        correctness = 0.6 * (matched_positions / len(optimal)) + 0.2 * (
            unique_critical_hits / len(optimal)
        )
        efficiency = 0.15 * min(1.0, len(optimal) / len(actions))
        safety = max(0.0, 0.05 - 0.03 * harmful_count - 0.01 * duplicate_count)

        return round(max(0.0, min(1.0, correctness + efficiency + safety)), 4)

    def _normalize(self, actions: Iterable[ActionType | str]) -> list[ActionType]:
        return [action if isinstance(action, ActionType) else ActionType(action) for action in actions]
