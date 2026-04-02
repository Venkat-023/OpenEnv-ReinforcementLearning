from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from environment.env import EmergencyFirstResponseDecisionEngine
from environment.models import Action, ResetRequest, StepOutput
from environment.tasks import TASKS


app = FastAPI(title="Emergency First-Response Decision Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
environment = EmergencyFirstResponseDecisionEngine()
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state")
def get_state() -> dict[str, object]:
    return environment.state().model_dump()


@app.get("/", response_model=None)
def root() -> Response:
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse(
        {
            "name": "Emergency First-Response Decision Engine",
            "status": "running",
            "docs": "/docs",
            "health": "/healthz",
        }
    )


@app.get("/assets/{asset_path:path}")
def frontend_assets(asset_path: str) -> FileResponse:
    asset_file = FRONTEND_DIST / "assets" / asset_path
    return FileResponse(asset_file)


@app.get("/tasks")
def list_tasks() -> list[dict[str, object]]:
    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty.value,
            "description": task.description,
            "scenario_summary": task.scenario_summary,
            "optimal_sequence": [action.value for action in task.optimal_sequence],
            "max_steps": task.max_steps,
        }
        for task in TASKS.values()
    ]


@app.post("/reset", response_model=StepOutput)
def reset_environment(payload: ResetRequest | None = Body(default=None)) -> StepOutput:
    try:
        observation = environment.reset(task_id=payload.task_id if payload else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepOutput(
        observation=observation,
        reward=0.0,
        done=False,
        info={
            "task_id": observation.task_id,
            "difficulty": observation.difficulty.value,
            "message": "Environment reset successfully.",
            "reward_signal": None,
        },
    )


@app.post("/step", response_model=StepOutput)
def step_environment(action: Action) -> StepOutput:
    observation, reward, done, info = environment.step(action)
    return StepOutput(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )
