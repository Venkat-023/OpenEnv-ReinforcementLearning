import { useEffect, useMemo, useState } from "react";

const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin);

const FALLBACK_TASKS = [
  {
    task_id: "cardiac_arrest_easy",
    difficulty: "easy",
    description: "Clear cardiac arrest case with nearby AED access.",
    scenario_summary:
      "An adult has collapsed in a staffed airport terminal and is not breathing normally.",
    max_steps: 6,
    optimal_sequence: [
      "CALL_EMERGENCY",
      "CHECK_BREATHING",
      "START_CPR",
      "USE_AED",
      "MONITOR_PATIENT"
    ]
  },
  {
    task_id: "severe_bleeding_medium",
    difficulty: "medium",
    description: "Severe external bleeding with partial physiological visibility.",
    scenario_summary:
      "A kitchen worker has a deep laceration with rapid blood loss and limited support.",
    max_steps: 7,
    optimal_sequence: [
      "CHECK_SCENE_SAFETY",
      "CALL_EMERGENCY",
      "APPLY_PRESSURE",
      "CHECK_PULSE",
      "MONITOR_PATIENT"
    ]
  },
  {
    task_id: "road_accident_hard",
    difficulty: "hard",
    description: "Deceptive trauma where hidden shock is masked by a minor visible wound.",
    scenario_summary:
      "A crash victim appears to have only minor visible bleeding, but the true danger is occult shock.",
    max_steps: 9,
    optimal_sequence: [
      "CHECK_SCENE_SAFETY",
      "CHECK_BREATHING",
      "CONTROL_AIRWAY",
      "CALL_EMERGENCY",
      "CHECK_PULSE",
      "APPLY_PRESSURE",
      "MONITOR_PATIENT"
    ]
  }
];

const ACTIONS = [
  "CALL_EMERGENCY",
  "CHECK_SCENE_SAFETY",
  "CHECK_RESPONSIVENESS",
  "CHECK_BREATHING",
  "CHECK_PULSE",
  "START_CPR",
  "USE_AED",
  "APPLY_PRESSURE",
  "CONTROL_AIRWAY",
  "PLACE_RECOVERY_POSITION",
  "MONITOR_PATIENT",
  "WAIT"
];

const ACTION_DETAILS = {
  CALL_EMERGENCY: {
    title: "Call Emergency",
    description: "Escalate to EMS or emergency support early in time-critical cases.",
    tone: "intervene"
  },
  CHECK_SCENE_SAFETY: {
    title: "Check Scene Safety",
    description: "Assess hazards before direct patient contact.",
    tone: "assess"
  },
  CHECK_RESPONSIVENESS: {
    title: "Check Responsiveness",
    description: "Establish whether the patient responds to voice or touch.",
    tone: "assess"
  },
  CHECK_BREATHING: {
    title: "Check Breathing",
    description: "Assess breathing and help reveal airway status.",
    tone: "assess"
  },
  CHECK_PULSE: {
    title: "Check Pulse",
    description: "Assess circulation and detect shock or arrest physiology.",
    tone: "assess"
  },
  START_CPR: {
    title: "Start CPR",
    description: "Begin compressions when pulseless apnea is confirmed.",
    tone: "intervene"
  },
  USE_AED: {
    title: "Use AED",
    description: "Apply defibrillation when arrest is confirmed and sequence is appropriate.",
    tone: "intervene"
  },
  APPLY_PRESSURE: {
    title: "Apply Pressure",
    description: "Control major external bleeding with direct pressure.",
    tone: "intervene"
  },
  CONTROL_AIRWAY: {
    title: "Control Airway",
    description: "Support a compromised airway and improve ventilation.",
    tone: "intervene"
  },
  PLACE_RECOVERY_POSITION: {
    title: "Recovery Position",
    description: "Protect the airway in selected unconscious-but-breathing cases.",
    tone: "intervene"
  },
  MONITOR_PATIENT: {
    title: "Monitor Patient",
    description: "Reassess after important actions and watch for change.",
    tone: "monitor"
  },
  WAIT: {
    title: "Wait",
    description: "Take no intervention for this step.",
    tone: "monitor"
  }
};

function App() {
  const [apiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [tasks, setTasks] = useState(FALLBACK_TASKS);
  const [selectedTaskId, setSelectedTaskId] = useState(FALLBACK_TASKS[0].task_id);
  const [selectedAction, setSelectedAction] = useState("CALL_EMERGENCY");
  const [queuedActions, setQueuedActions] = useState([]);
  const [stepOutput, setStepOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasStarted, setHasStarted] = useState(false);

  useEffect(() => {
    async function loadTasks() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(`${apiBaseUrl}/tasks`);
        if (!response.ok) {
          throw new Error(`Unable to load tasks (${response.status})`);
        }

        const payload = await response.json();
        if (Array.isArray(payload) && payload.length > 0) {
          setTasks(payload);
          setSelectedTaskId(payload[0].task_id);
        }
      } catch (err) {
        setTasks(FALLBACK_TASKS);
        setError(`${err.message || "Unable to load tasks."} Using built-in task list instead.`);
      } finally {
        setLoading(false);
      }
    }

    loadTasks();
  }, [apiBaseUrl]);

  async function handleReset(taskId = selectedTaskId) {
    if (!taskId) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${apiBaseUrl}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId })
      });

      if (!response.ok) {
        throw new Error(`Reset failed (${response.status})`);
      }

      const payload = await response.json();
      setStepOutput(payload);
      setSelectedTaskId(taskId);
      setHasStarted(true);
      setQueuedActions([]);
      setSelectedAction(payload.observation.available_actions?.[0] || "CALL_EMERGENCY");
    } catch (err) {
      setError(err.message || "Reset failed.");
    } finally {
      setLoading(false);
    }
  }

  async function submitAction(actionType) {
    const response = await fetch(`${apiBaseUrl}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_type: actionType })
    });

    if (!response.ok) {
      throw new Error(`Step failed (${response.status})`);
    }

    const payload = await response.json();
    setStepOutput(payload);
    setHasStarted(true);
    return payload;
  }

  async function handleStep() {
    if (!selectedAction) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      await submitAction(selectedAction);
    } catch (err) {
      setError(err.message || "Step failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRunQueue() {
    if (!queuedActions.length) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      let latestPayload = null;
      let remainingQueue = [...queuedActions];

      while (remainingQueue.length) {
        const nextAction = remainingQueue[0];
        latestPayload = await submitAction(nextAction);
        remainingQueue = remainingQueue.slice(1);
        setQueuedActions(remainingQueue);
        if (latestPayload.done) {
          break;
        }
      }
    } catch (err) {
      setError(err.message || "Queue execution failed.");
    } finally {
      setLoading(false);
    }
  }

  const selectedTask = tasks.find((task) => task.task_id === selectedTaskId) || FALLBACK_TASKS[0];
  const observation = stepOutput?.observation;
  const info = stepOutput?.info;
  const rewardSignal = info?.reward_signal;
  const actionsTaken = observation?.actions_taken || [];
  const availableActions = observation?.available_actions || ACTIONS;

  useEffect(() => {
    if (!availableActions.includes(selectedAction)) {
      setSelectedAction(availableActions[0] || "");
    }

    setQueuedActions((current) => current.filter((action) => availableActions.includes(action)));
  }, [availableActions, selectedAction]);

  const actionCards = useMemo(() => {
    return availableActions.map((action) => ({
      action,
      ...ACTION_DETAILS[action]
    }));
  }, [availableActions]);

  const environmentContext =
    observation && typeof observation.environment_context === "object"
      ? observation.environment_context
      : null;

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Decision Support UI</p>
          <h1>Emergency First-Response Decision Engine</h1>
          <p className="hero-copy">
            Build a response sequence from the available actions, then submit one step or run a short queue.
          </p>
        </div>
        <div className="api-pill">Run everything from this page</div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>Scenario Control</h2>
          <p className="muted">Choose the case you want to run.</p>

          <div className="task-list">
            {tasks.map((task) => (
              <button
                key={task.task_id}
                type="button"
                className={`task-card selectable ${selectedTaskId === task.task_id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedTaskId(task.task_id);
                  setStepOutput(null);
                  setHasStarted(false);
                  setQueuedActions([]);
                  setError("");
                }}
              >
                <div className="tag-row">
                  <span className={`tag tag-${task.difficulty}`}>{task.difficulty}</span>
                  <span className="tag">max steps {task.max_steps}</span>
                </div>
                <strong>{task.task_id}</strong>
                <p>{task.description}</p>
                <p className="muted">{task.scenario_summary}</p>
              </button>
            ))}
          </div>

          <div className="button-row">
            <button className="primary-button" onClick={() => handleReset(selectedTaskId)} disabled={loading || !selectedTaskId}>
              {loading ? "Working..." : hasStarted ? "Reset Current Scenario" : "Start Selected Scenario"}
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>Action Panel</h2>
          <p className="muted">
            Click an action to select it for a single step, or add several actions into the queue and run them in order.
          </p>

          <div className="action-toolbar">
            <div className="toolbar-card">
              <span className="status-label">Selected Action</span>
              <strong>{selectedAction ? labelize(selectedAction) : "None"}</strong>
            </div>
            <div className="toolbar-card">
              <span className="status-label">Queued Actions</span>
              <strong>{queuedActions.length}</strong>
            </div>
          </div>

          <div className="action-library">
            {actionCards.map(({ action, title, description, tone }) => (
              <button
                key={action}
                type="button"
                className={`action-card action-tone-${tone} ${selectedAction === action ? "selected" : ""}`}
                onClick={() => setSelectedAction(action)}
              >
                <div className="action-card-head">
                  <strong>{title}</strong>
                  <span className="action-code">{action}</span>
                </div>
                <p>{description}</p>
                <div className="action-card-foot">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={(event) => {
                      event.stopPropagation();
                      setQueuedActions((current) => [...current, action]);
                    }}
                  >
                    Add to Queue
                  </button>
                  {actionsTaken.includes(action) ? <span className="mini-tag mini-tag-used">used</span> : null}
                </div>
              </button>
            ))}
          </div>

          <div className="queue-panel">
            <div className="subpanel-header">
              <h3>Action Queue</h3>
              <button
                type="button"
                className="text-button"
                onClick={() => setQueuedActions([])}
                disabled={!queuedActions.length || loading}
              >
                Clear Queue
              </button>
            </div>

            {queuedActions.length ? (
              <div className="queue-list">
                {queuedActions.map((action, index) => (
                  <div className="queue-item" key={`${action}-${index}`}>
                    <span>{index + 1}. {labelize(action)}</span>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() =>
                        setQueuedActions((current) => current.filter((_, itemIndex) => itemIndex !== index))
                      }
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No actions queued yet.</p>
            )}
          </div>

          <div className="button-cluster">
            <button className="primary-button" onClick={handleStep} disabled={loading || !stepOutput || !selectedAction}>
              Submit Selected Action
            </button>
            <button className="secondary-button" onClick={handleRunQueue} disabled={loading || !stepOutput || !queuedActions.length}>
              Run Action Queue
            </button>
          </div>

          <div className="status-strip reward-strip">
            <MetricCard label="Action Reward" value={rewardSignal?.action_reward ?? "-"} />
            <MetricCard label="Time Decay" value={rewardSignal?.time_decay_factor ?? "-"} />
            <MetricCard label="Total Reward" value={rewardSignal?.total ?? stepOutput?.reward ?? "-"} />
            <MetricCard label="Score" value={info?.score_so_far ?? "-"} />
          </div>

          <div className="reward-reason">
            <span className="status-label">Why This Reward Happened</span>
            <p>{rewardSignal?.reason || "Take a step to see the clinical explanation for the reward."}</p>
          </div>

          {error ? <p className="error-banner">{error}</p> : null}
          {!hasStarted ? (
            <p className="muted">Start a scenario first, then choose one or more actions.</p>
          ) : null}
        </section>

        <section className="panel">
          <h2>Observation</h2>
          {observation ? (
            <div className="details-grid">
              <Detail label="Risk Level" value={observation.risk_level} />
              <Detail label="Time Elapsed" value={observation.time_elapsed} />
              <Detail label="Consciousness" value={observation.patient_condition.conscious_status} />
              <Detail label="Breathing" value={observation.patient_condition.breathing_status} />
              <Detail label="Bleeding" value={observation.patient_condition.bleeding_severity} />
              <Detail label="Pulse" value={observation.patient_condition.pulse_status} />
              <Detail label="Airway" value={observation.patient_condition.airway_status} />
              <Detail label="Location" value={environmentContext?.location_type || "unknown"} />
              <Detail label="Help" value={environmentContext?.help_availability || "unknown"} />
            </div>
          ) : (
            <p className="muted">Reset a task to load the first observation.</p>
          )}
        </section>

        <section className="panel">
          <h2>Scenario Guidance</h2>
          <p>{selectedTask?.scenario_summary}</p>

          <h3>Expected Clinical Flow</h3>
          <div className="chip-wrap">
            {(selectedTask?.optimal_sequence || []).map((action, index) => (
              <span className="chip" key={`${action}-${index}`}>
                {index + 1}. {labelize(action)}
              </span>
            ))}
          </div>

          <h3>Actions Taken</h3>
          <div className="chip-wrap">
            {(actionsTaken || []).map((action, index) => (
              <span className="chip chip-history" key={`${action}-${index}`}>
                {index + 1}. {labelize(action)}
              </span>
            ))}
          </div>

          <h3>Termination</h3>
          <p className="muted">{info?.termination_reason || "Episode is active."}</p>
        </section>

        <section className="panel">
          <h2>Structured JSON</h2>
          <pre>{JSON.stringify(stepOutput, null, 2) || "{}"}</pre>
        </section>
      </main>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div>
      <span className="status-label">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function labelize(value) {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default App;
