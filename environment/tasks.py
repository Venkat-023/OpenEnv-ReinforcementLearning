from __future__ import annotations

from environment.models import (
    ActionType,
    AirwayStatus,
    BleedingSeverity,
    BreathingStatus,
    ConsciousStatus,
    DifficultyLevel,
    EnvironmentContext,
    PatientCondition,
    PulseStatus,
    TaskDefinition,
)


CARDIAC_ARREST_EASY = TaskDefinition(
    task_id="cardiac_arrest_easy",
    difficulty=DifficultyLevel.EASY,
    description="Clear cardiac arrest scenario in a public location with strong cues and nearby equipment.",
    scenario_summary=(
        "An adult has collapsed in a staffed airport terminal. A bystander reports the patient is "
        "unresponsive and not breathing normally. An AED cabinet is visible nearby."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNCONSCIOUS,
        breathing_status=BreathingStatus.ABSENT,
        bleeding_severity=BleedingSeverity.NONE,
        pulse_status=PulseStatus.ABSENT,
        airway_status=AirwayStatus.CLEAR,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNKNOWN,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.UNKNOWN,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="airport_terminal",
        help_availability="trained staff and bystanders nearby",
        hazards=[],
        equipment_available=["AED", "first_aid_kit", "phone"],
    ),
    max_steps=6,
    optimal_sequence=[
        ActionType.CALL_EMERGENCY,
        ActionType.CHECK_BREATHING,
        ActionType.START_CPR,
        ActionType.USE_AED,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Emergency services contacted, CPR started promptly, and AED applied.",
    failure_criteria="Repeated delays in CPR or AED use cause irreversible deterioration.",
    hidden_notes={"patient_type": "adult"},
)


SEVERE_BLEEDING_MEDIUM = TaskDefinition(
    task_id="severe_bleeding_medium",
    difficulty=DifficultyLevel.MEDIUM,
    description="Visible severe bleeding with partial physiological information and required prioritization.",
    scenario_summary=(
        "A kitchen worker has a deep laceration to the forearm from broken glass. Blood is rapidly soaking "
        "through towels. The patient is awake but frightened, and professional help is not yet on scene."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.ALERT,
        breathing_status=BreathingStatus.NORMAL,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.CLEAR,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNKNOWN,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.SEVERE,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="restaurant_kitchen",
        help_availability="one coworker available to assist",
        hazards=["broken_glass", "slippery_floor"],
        equipment_available=["gloves", "clean_cloth", "phone"],
    ),
    max_steps=7,
    optimal_sequence=[
        ActionType.CHECK_SCENE_SAFETY,
        ActionType.CALL_EMERGENCY,
        ActionType.APPLY_PRESSURE,
        ActionType.CHECK_PULSE,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Scene safety addressed, bleeding controlled, pulse reassessed, and patient monitored.",
    failure_criteria="Bleeding remains uncontrolled long enough to cause shock and collapse.",
    hidden_notes={"injury_location": "forearm"},
)


ROAD_ACCIDENT_HARD = TaskDefinition(
    task_id="road_accident_hard",
    difficulty=DifficultyLevel.HARD,
    description="Deceptive roadside trauma where minor visible bleeding hides evolving internal hemorrhagic shock.",
    scenario_summary=(
        "A crash victim is found beside the road after a low-speed collision. There is only a small amount of visible "
        "bleeding from the abdomen, but the patient is unconscious with shallow breathing and cool pale skin. Traffic is "
        "still passing nearby and there is no immediate medical support."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNCONSCIOUS,
        breathing_status=BreathingStatus.SHALLOW,
        bleeding_severity=BleedingSeverity.CRITICAL,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.COMPROMISED,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNKNOWN,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.UNKNOWN,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="roadside",
        help_availability="one untrained driver stopped to help",
        hazards=["moving_traffic", "fuel_spill_risk"],
        equipment_available=["phone", "clean_cloth", "high_visibility_jacket"],
    ),
    max_steps=9,
    optimal_sequence=[
        ActionType.CHECK_SCENE_SAFETY,
        ActionType.CHECK_BREATHING,
        ActionType.CONTROL_AIRWAY,
        ActionType.CALL_EMERGENCY,
        ActionType.CHECK_PULSE,
        ActionType.APPLY_PRESSURE,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="The agent works through ABC assessment, detects shock physiology, and stabilizes airway and circulation.",
    failure_criteria="Superficial treatment or skipped circulation assessment allows hidden hemorrhagic shock to progress.",
    hidden_notes={
        "injury_location": "abdomen",
        "hidden_condition": "internal_bleeding_hypovolemic_shock",
        "visible_wound": "minor_abdominal_bleeding",
    },
)


ANAPHYLAXIS_MEDIUM = TaskDefinition(
    task_id="anaphylaxis_medium",
    difficulty=DifficultyLevel.MEDIUM,
    description="Rapid allergic airway emergency requiring early breathing assessment and airway support.",
    scenario_summary=(
        "A 28-year-old has a severe allergic reaction after eating at a restaurant. She has hives, facial swelling, "
        "difficulty breathing, and appears lightheaded with dropping blood pressure. An EpiPen is reported to be in her bag."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.CONFUSED,
        breathing_status=BreathingStatus.SHALLOW,
        bleeding_severity=BleedingSeverity.NONE,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.COMPROMISED,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNKNOWN,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.UNKNOWN,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="restaurant_dining_room",
        help_availability="bystanders and staff nearby",
        hazards=[],
        equipment_available=["phone", "first_aid_kit", "patient_epipen"],
    ),
    max_steps=12,
    optimal_sequence=[
        ActionType.CHECK_SCENE_SAFETY,
        ActionType.CALL_EMERGENCY,
        ActionType.CHECK_BREATHING,
        ActionType.CONTROL_AIRWAY,
        ActionType.CHECK_PULSE,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Airway risk recognized early, emergency help activated, and respiratory status monitored after intervention.",
    failure_criteria="Delayed airway support allows progressive airway obstruction and cardiovascular collapse.",
    hidden_notes={"trigger": "food_allergy"},
)


CHOKING_EASY = TaskDefinition(
    task_id="choking_easy",
    difficulty=DifficultyLevel.EASY,
    description="Obvious upper-airway obstruction where delay is especially dangerous.",
    scenario_summary=(
        "An elderly man at a restaurant is clutching his throat, cannot speak or cough, and his face is turning red. "
        "No breathing sounds are audible and bystanders are present."
    ),
    initial_true_condition=PatientCondition(
        conscious_status=ConsciousStatus.ALERT,
        breathing_status=BreathingStatus.ABSENT,
        bleeding_severity=BleedingSeverity.NONE,
        pulse_status=PulseStatus.WEAK,
        airway_status=AirwayStatus.COMPROMISED,
    ),
    initial_observed_condition=PatientCondition(
        conscious_status=ConsciousStatus.UNKNOWN,
        breathing_status=BreathingStatus.UNKNOWN,
        bleeding_severity=BleedingSeverity.UNKNOWN,
        pulse_status=PulseStatus.UNKNOWN,
        airway_status=AirwayStatus.UNKNOWN,
    ),
    environment_context=EnvironmentContext(
        location_type="restaurant_dining_room",
        help_availability="multiple bystanders available",
        hazards=[],
        equipment_available=["phone"],
    ),
    max_steps=8,
    optimal_sequence=[
        ActionType.CHECK_RESPONSIVENESS,
        ActionType.CALL_EMERGENCY,
        ActionType.CONTROL_AIRWAY,
        ActionType.CHECK_BREATHING,
        ActionType.MONITOR_PATIENT,
    ],
    success_criteria="Airway obstruction relieved early and the patient remains under observation while help is on the way.",
    failure_criteria="Waiting or delayed airway management leads to unconsciousness and arrest.",
    hidden_notes={"obstruction": "foreign_body"},
)


TASKS: dict[str, TaskDefinition] = {
    CARDIAC_ARREST_EASY.task_id: CARDIAC_ARREST_EASY,
    SEVERE_BLEEDING_MEDIUM.task_id: SEVERE_BLEEDING_MEDIUM,
    ROAD_ACCIDENT_HARD.task_id: ROAD_ACCIDENT_HARD,
    ANAPHYLAXIS_MEDIUM.task_id: ANAPHYLAXIS_MEDIUM,
    CHOKING_EASY.task_id: CHOKING_EASY,
}


DEFAULT_TASK_ID = CARDIAC_ARREST_EASY.task_id
