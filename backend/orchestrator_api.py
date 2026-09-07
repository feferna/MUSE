import os
import csv
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, Form, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import yaml

from orchestrator import OptimizationOrchestrator

# Grab the current directory where we're running from
current_folder = os.path.abspath(os.getcwd())

# Set up a logs folder for API logs under the new run tree
SERVER_LOGS = os.path.join(current_folder, "run", "_server", "logs")
os.makedirs(SERVER_LOGS, exist_ok=True)

# Create a log file named with the current timestamp
log_file_name = os.path.join(
    SERVER_LOGS, f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

app = FastAPI()
RUN_ROOT = os.path.join(current_folder, "run")
os.makedirs(RUN_ROOT, exist_ok=True)  # ensure it exists at startup
app.mount("/run", StaticFiles(directory=RUN_ROOT), name="run")

orchestrator = OptimizationOrchestrator()


def _toggle_mode_from_orchestrator(orchestrator, default_mode: str) -> str:
    try:
        combo = getattr(orchestrator, "sources_to_use", None)
        if combo:
            return "+".join(combo)
    except Exception:
        pass
    return default_mode or "tool"


def _norm_source(s):
    if not s:
        return None
    s = str(s).strip().lower()
    if s in ("llm", "sim", "simulation", "model"):
        return "llm"
    if s in ("human", "pref", "preference", "user"):
        return "human"
    if s in ("wave", "color", "colour"):
        return "wave"
    return None


def _toggle_mode_from_sources(body: dict) -> str | None:
    """
    Build the canonical toggle mode strictly from source1/2/3 in the payload.
    Example: 'llm+human', 'llm', 'wave', etc.
    Returns None if no sources are provided.
    """
    srcs: list[str] = []
    for key in ("source1", "source2", "source3"):
        n = _norm_source(body.get(key))
        if n and n not in srcs:
            srcs.append(n)
    return "+".join(srcs) if srcs else None


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

# Set up CORS so our frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-User-Id", "X-Session-Id"],
)

# Where we'll store user-uploaded config files
CONFIG_DIRECTORY = os.path.abspath("./configs")
os.makedirs(CONFIG_DIRECTORY, exist_ok=True)


def _get_user_id(request: Request, fallback: str | None = None) -> str:
    return (
        request.headers.get("X-User-Id")
        or request.query_params.get("uid")
        or fallback
        or "anon"
    )


def _get_session_id(request: Request) -> str:
    return request.headers.get("X-Session-Id") or ""


@app.get("/status")
async def get_status():
    """
    Basic health check endpoint to verify server is running.
    """
    return {"status": "ok", "message": "Server is running"}


@app.post("/log")
async def log_event(request: Request):
    """
    Log events from the frontend for analytics and debugging.

    Guard: only write after a task is selected.
    Folder routing:
      - chart_reader -> run/<user>/chartist_task/<chart_mode>/<chart_task>/frontend_logs
      - toggle_task/design -> run/<user>/toggle_task/<toggle_mode>/frontend_logs
      - tutorial/byos -> run/<user>/<task_name>/frontend_logs
    """
    try:
        body = await request.json()
        event_type = body.get("eventType")
        detail = body.get("detail")
        path = body.get("path") or ""
        if not event_type:
            raise HTTPException(
                status_code=400, detail="Missing 'eventType' in the request body.")

        # ---------- GUARD + STUDY FOLDER ----------
        task_name = (body.get("task_name") or "").strip()
        if not task_name:
            return Response(status_code=204)

        if task_name == "chart_reader":
            chart_mode = (body.get("chart_mode") or "").strip()
            chart_task = (body.get("chart_task") or "").strip()
            if not chart_mode or not chart_task:
                return Response(status_code=204)
            study = f"chartist_task/{chart_mode}/{chart_task}"

        elif task_name in {"toggle_task", "toggle_design"}:
            toggle_mode = _toggle_mode_from_sources(body)  # <-- strict: only from source1/2/3
            if not toggle_mode:
                return Response(status_code=204)  # nothing to log without sources
            study = f"toggle_task/{toggle_mode}"

        elif task_name in {"tutorial", "byos"}:
            study = task_name
        else:
            return Response(status_code=204)
        # -----------------------------------------

        user_id = _get_user_id(request)
        session_id = _get_session_id(request)
        ts = datetime.now().isoformat()

        frontend_logs_folder = os.path.join(
            current_folder, "run", user_id, study, "frontend_logs")
        os.makedirs(frontend_logs_folder, exist_ok=True)

        csv_file_path = os.path.join(
            frontend_logs_folder, "frontend_events.csv")
        file_exists = os.path.isfile(csv_file_path)

        # ---------- TASK-SPECIFIC SCHEMA ----------
        if task_name == "chart_reader":
            header = [
                "timestamp", "user_id", "session_id", "study_folder",
                "task_name", "chart_mode", "chart_task",
                "event_type", "detail", "path",
            ]
            row = [
                ts, user_id, session_id, study,
                task_name, chart_mode, chart_task,
                event_type, detail, path,
            ]
        elif task_name == "toggle_task":
            header = [
                "timestamp", "user_id", "session_id", "study_folder",
                "task_name", "toggle_mode",
                "event_type", "detail", "path",
            ]
            row = [
                ts, user_id, session_id, study,
                task_name, toggle_mode,   # use the canonical string from sources
                event_type, detail, path,
            ]
        else:  # tutorial/byos
            header = [
                "timestamp", "user_id", "session_id", "study_folder",
                "task_name",
                "event_type", "detail", "path",
            ]
            row = [
                ts, user_id, session_id, study,
                task_name,
                event_type, detail, path,
            ]
        # ------------------------------------------

        with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

        return {"status": "Event logged successfully."}

    except Exception as e:
        logging.error(f"ERROR in /log endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_user_files")
async def create_env_file(
    request: Request,
    config_yaml: str = Form(...),
    environment: UploadFile = Form(...),
    user_evaluation: UploadFile = Form(...),
    user_id: str | None = Form(None),
):
    try:
        uid = user_id or _get_user_id(request)

        # Parse + normalize config; ensure participant_id and derived study folder
        try:
            cfg = yaml.safe_load(config_yaml) or {}
            conf = cfg.setdefault("configuration", {})
            if not conf.get("participant_id"):
                conf["participant_id"] = uid

            def _norm_source(s):
                if not s:
                    return None
                s = str(s).strip().lower()
                if s in ("llm", "sim", "simulation", "model"):
                    return "llm"
                if s in ("human", "pref", "preference", "user"):
                    return "human"
                if s in ("wave", "color", "colour"):
                    return "wave"
                return None

            task_name = (conf.get("task_name") or "").strip()
            chart_mode = (conf.get("chart_mode") or "").strip()
            chart_task = (conf.get("chart_task") or "").strip()
            toggle_mode = (conf.get("toggle_mode") or "").strip()
            fallback_study = (conf.get("study_name") or "").strip()

            # Prefer orchestrator’s active sources (handles forced ["llm","human","wave"])
            if task_name in {"toggle_task", "toggle_design"}:
                toggle_mode = _toggle_mode_from_orchestrator(
                    orchestrator, toggle_mode)
                conf["toggle_mode"] = toggle_mode

            if task_name in {"toggle_task", "toggle_design"}:
                # build ordered, unique list from source1..3 (from either root or configuration level)
                srcs_raw = []
                for key in ("source1", "source2", "source3"):
                    v = (cfg.get(key) or conf.get(key))
                    n = _norm_source(v)
                    if n and n not in srcs_raw:
                        srcs_raw.append(n)
                if srcs_raw:
                    toggle_mode = "+".join(srcs_raw)
                    # normalize into YAML so orchestrator and API agree
                    conf["toggle_mode"] = toggle_mode

            # Derive the canonical per-task study folder
            if task_name == "chart_reader" and chart_mode and chart_task:
                study_folder = f"chartist_task/{chart_mode}/{chart_task}"
            elif task_name in {"toggle_task", "toggle_design"} and toggle_mode:
                study_folder = f"toggle_task/{toggle_mode}"
            elif task_name in {"tutorial", "byos"}:
                study_folder = task_name
            else:
                # Last resort: keep whatever was passed, or default
                study_folder = fallback_study or task_name or "default"

            # Normalize the study_name inside the YAML so the orchestrator uses it too
            conf["study_name"] = study_folder

            # Write back the patched YAML
            config_yaml = yaml.safe_dump(cfg, sort_keys=False)
        except Exception as e:
            logging.warning(f"Could not normalize YAML: {e}")
            # Fall back to a safe place
            study_folder = "default"

        # --- 1) Runtime copy (orchestrator reads this) ---
        config_file_path = os.path.join(CONFIG_DIRECTORY, "config.yaml")
        with open(config_file_path, "w") as f:
            f.write(config_yaml)

        # --- 2) Archival copy in run/<user>/<derived study folder>/ ---
        run_cfg_dir = os.path.join(current_folder, "run", uid, study_folder)
        os.makedirs(run_cfg_dir, exist_ok=True)
        run_config_file_path = os.path.join(run_cfg_dir, "config.yaml")
        with open(run_config_file_path, "w") as f:
            f.write(config_yaml)

        # Save environment & evaluation (kept under ./configs for imports)
        env_file_path = os.path.join(CONFIG_DIRECTORY, environment.filename)
        with open(env_file_path, "wb") as f:
            f.write(await environment.read())

        eval_file_path = os.path.join(
            CONFIG_DIRECTORY, user_evaluation.filename)
        with open(eval_file_path, "wb") as f:
            f.write(await user_evaluation.read())

        return {
            "status": "Files uploaded successfully",
            "config_file_path": config_file_path,          # runtime
            # archival (derived path)
            "run_config_file_path": run_config_file_path,
            "env_file_path": env_file_path,
            "eval_file_path": eval_file_path,
            "user_id": uid,
            "study_name": study_folder,                    # normalized name returned
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})


@app.post("/start_optimization")
async def start_optimization(request: Request):
    # Fire up the optimization process
    try:
        uid = _get_user_id(request)
        # Attach to orchestrator; orchestrator can use it for per-user folders
        setattr(orchestrator, "user_id", uid)
        os.environ["PARTICIPANT_ID"] = uid  # also visible to any subprocess

        # Tell the orchestrator to get started with the optimization
        response = await orchestrator.start_optimization()
        return {"status": "Optimization started", "details": response}
    except Exception as e:
        logging.error("ERROR in /start_optimization: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization_status")
async def get_optimization_status(request: Request):
    try:
        uid = _get_user_id(request)
        setattr(orchestrator, "user_id", uid)

        status, completed_trials = orchestrator.get_optimization_status()
        return {
            "status": status,
            "completed_trials": completed_trials,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/best_trial")
async def get_best_trial(request: Request):
    try:
        uid = _get_user_id(request)
        setattr(orchestrator, "user_id", uid)

        payload = orchestrator.get_best_trial()  # expects a dict with video_url

        if not isinstance(payload, dict) or payload.get("best_value") is None:
            raise HTTPException(status_code=404, detail="No best trial available")

        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/current_batch_results")
async def get_current_batch_results(request: Request):
    """
    Get the latest on the current batch of trials:
    - If batch is still running: you'll get a status update
    - If batch is done: you'll get all trial parameters and results
    - If no batch started: you'll get a message saying so
    """
    uid = _get_user_id(request)
    setattr(orchestrator, "user_id", uid)
    batch_info = orchestrator.get_current_batch_info()
    return batch_info


@app.post("/choose_trial")
async def choose_trial(request: Request):
    """
    Let the user tell us which trials they preferred from the current batch.

    Expected format:
    {
        "chosen_trials": [5, 7],       // The ones they liked
        "not_chosen_trials": [2, 3, 4] // The ones they didn't like as much
    }
    """
    try:
        body = await request.json()
        chosen_trials = body.get("chosen_trials")
        not_chosen_trials = body.get("not_chosen_trials")

        # Make sure chosen_trials is a valid list of positive numbers
        if (not isinstance(chosen_trials, list) or
                not all(isinstance(x, int) and x >= 0 for x in chosen_trials)):
            raise ValueError(
                "Invalid 'chosen_trials'. Must be a list of non-negative integers.")

        # Same check for not_chosen_trials
        if (not isinstance(not_chosen_trials, list) or
                not all(isinstance(x, int) and x >= 0 for x in not_chosen_trials)):
            raise ValueError(
                "Invalid 'not_chosen_trials'. Must be a list of non-negative integers.")

        # Tell the orchestrator about the user's preferences
        setattr(orchestrator, "user_id", _get_user_id(request))

        orchestrator.chosen_trials = chosen_trials
        orchestrator.not_chosen_trials = not_chosen_trials

        return {"message": f"Received chosen_trials: {chosen_trials} and not_chosen_trials: {not_chosen_trials}"}

    except (ValueError, TypeError) as e:
        return {"error": str(e)}


@app.post("/stop_optimization")
async def stop_optimization_endpoint(request: Request):
    global orchestrator
    try:
        uid = _get_user_id(request)
        setattr(orchestrator, "user_id", uid)

        # Halt the optimization process
        await orchestrator.stop_optimization()

        return {"status": "Optimization stopped and orchestrator reset successfully."}
    except Exception as e:
        logging.error("ERROR in /stop_optimization: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disagreement_data")
async def get_disagreement_data():
    """
    Get data about where our models disagree with each other and how uncertain they are.
    Useful for understanding conflicting objectives and exploring the design space.
    """
    try:
        # Package up both disagreement and uncertainty into one response
        disagreement_data = {
            "disagreement": orchestrator.disagreement_data,
            "uncertainty": orchestrator.uncertainty_data
        }
        return disagreement_data
    except Exception as e:
        logging.error("ERROR in /disagreement_data: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug_state")
async def debug_state(request: Request):
    """
    Return a compact snapshot of the orchestrator state useful for debugging
    source selection and which prediction dicts are available.
    """
    try:
        uid = _get_user_id(request)
        setattr(orchestrator, "user_id", uid)

        def _keys_or_empty(d):
            try:
                return list(d.keys()) if d is not None else []
            except Exception:
                return []

        payload = {
            "user_id": getattr(orchestrator, "user_id", None),
            "sources_to_use": getattr(orchestrator, "sources_to_use", None),
            "sources_explicit": getattr(orchestrator, "sources_explicit", False),
            "toggle_mode": getattr(orchestrator, "toggle_mode", None),
            "r_predictions_keys": _keys_or_empty(getattr(orchestrator, "r_predictions", {})),
            "g_predictions_keys": _keys_or_empty(getattr(orchestrator, "g_predictions", {})),
            "w_predictions_keys": _keys_or_empty(getattr(orchestrator, "w_predictions", {})),
            "wave_metric_values_keys": _keys_or_empty(getattr(orchestrator, "wave_metric_values", {})),
            "uncertainty_keys": _keys_or_empty(getattr(orchestrator, "uncertainty_data", {})),
            "disagreement_keys": _keys_or_empty(getattr(orchestrator, "disagreement_data", {})),
        }

        return JSONResponse(status_code=200, content=payload)
    except Exception as e:
        logging.error(f"ERROR in /debug_state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.0", port=8001)
