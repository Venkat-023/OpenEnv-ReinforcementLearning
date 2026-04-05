from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConsciousStatus(str, Enum):
    ALERT = "alert"
    CONFUSED = "confused"
    UNCONSCIOUS = "unconscious"
    UNKNOWN = "unknown"


class BreathingStatus(str, Enum):
    NORMAL = "normal"
    SHALLOW = "shallow"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class BleedingSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PulseStatus(str, Enum):
    NORMAL = "normal"
    WEAK = "weak"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class AirwayStatus(str, Enum):
    CLEAR = "clear"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    CALL_EMERGENCY = "CALL_EMERGENCY"
    CHECK_SCENE_SAFETY = "CHECK_SCENE_SAFETY"
    CHECK_RESPONSIVENESS = "CHECK_RESPONSIVENESS"
    CHECK_BREATHING = "CHECK_BREATHING"
    CHECK_PULSE = "CHECK_PULSE"
    START_CPR = "START_CPR"
    USE_AED = "USE_AED"
    APPLY_PRESSURE = "APPLY_PRESSURE"
    CONTROL_AIRWAY = "CONTROL_AIRWAY"
    PLACE_RECOVERY_POSITION = "PLACE_RECOVERY_POSITION"
    MONITOR_PATIENT = "MONITOR_PATIENT"
    WAIT = "WAIT"


class PatientCondition(BaseModel):
    conscious_status: ConsciousStatus
    breathing_status: BreathingStatus
    bleeding_severity: BleedingSeverity
    pulse_status: PulseStatus
    airway_status: AirwayStatus


class EnvironmentContext(BaseModel):
    location_type: str
    help_availability: str
    hazards: list[str] = Field(default_factory=list)
    equipment_available: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    scenario_summary: str
    patient_condition: PatientCondition
    time_elapsed: int
    actions_taken: list[ActionType]
    environment_context: EnvironmentContext | str
    available_actions: list[ActionType]
    last_action_effect: str
    risk_level: str


class Action(BaseModel):
    action_type: ActionType


class StepOutput(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class RewardSignal(BaseModel):
    action_reward: float
    reason: str
    time_decay_factor: float
    total: float
    value: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    rationale: str | None = None


class ResetRequest(BaseModel):
    task_id: str | None = None


class InternalState(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    scenario_summary: str
    true_condition: PatientCondition
    observed_condition: PatientCondition
    environment_context: EnvironmentContext
    time_elapsed: int = 0
    max_steps: int
    actions_taken: list[ActionType] = Field(default_factory=list)
    emergency_called: bool = False
    cpr_started: bool = False
    aed_used: bool = False
    pressure_applied: bool = False
    airway_controlled: bool = False
    responsiveness_checked: bool = False
    breathing_checked: bool = False
    pulse_checked: bool = False
    steps_without_critical_action: int = 0
    steps_to_first_critical_action: int | None = None
    revealed_fields: set[str] = Field(default_factory=set)
    done: bool = False
    termination_reason: str | None = None
    success: bool = False
    last_action_effect: str = "Scenario initialized."
    optimal_sequence: list[ActionType] = Field(default_factory=list)
    hidden_notes: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    description: str
    scenario_summary: str
    initial_true_condition: PatientCondition
    initial_observed_condition: PatientCondition
    environment_context: EnvironmentContext
    max_steps: int
    optimal_sequence: list[ActionType]
    success_criteria: str
    failure_criteria: str
    hidden_notes: dict[str, Any] = Field(default_factory=dict)
