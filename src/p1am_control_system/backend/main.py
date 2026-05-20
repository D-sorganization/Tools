import asyncio
import csv
import io
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from alicat_manager import AlicatManager, AlicatMFC
from database import get_session, init_db
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from models import (
    AlicatGasPayload,
    AlicatMFCState,
    AlicatSetpointPayload,
    EventLog,
    InterlockConfig,
    PIDConfig,
    PIDTuningStepPayload,
    RoutingConfig,
    TagLog,
)
from plc_factory import PLCFactory
from pydantic import BaseModel
from pydantic import Field as PydanticField
from simulator_client import SimulatedPLCClient
from sqlmodel import Session, col, select
from tools_core import scada

AlarmEngine = scada.AlarmEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dcs_backend.main")

# Instantiate active PLC client and backup simulator for offline fallback
plc_client = PLCFactory.create_client()
modbus_manager = plc_client  # Compatibility alias
backup_simulator = SimulatedPLCClient()


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket client connections and broadcasts live updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket client connected.")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected.")

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to WebSocket client: {e}")


ws_manager = ConnectionManager()
shutdown_event = asyncio.Event()

# Instantiate global Alicat manager and default devices
alicat_manager = AlicatManager()
alicat_manager.add_device(
    AlicatMFC(
        device_id="A",
        name="Oxygen MFC",
        gas="O2",
        max_flow=50.0,
        connection_type="mock",
    )
)
alicat_manager.add_device(
    AlicatMFC(
        device_id="B",
        name="Nitrogen MFC",
        gas="N2",
        max_flow=100.0,
        connection_type="mock",
    )
)
alicat_manager.add_device(
    AlicatMFC(
        device_id="C",
        name="Carbon Dioxide MFC",
        gas="CO2",
        max_flow=20.0,
        connection_type="mock",
    )
)

latest_tags: list[float] = [0.0] * 32
active_config: RoutingConfig = RoutingConfig(
    input_routing=[0, 1, 2, 3, 4, 5],
    output_routing=[10, 11],
    pids=[
        PIDConfig(pv_tag_id=1, cv_tag_id=2, setpoint=50.0, kp=1.0, ki=0.5, kd=0.1),
        PIDConfig(pv_tag_id=3, cv_tag_id=4, setpoint=30.0, kp=1.5, ki=0.2, kd=0.05),
        PIDConfig(pv_tag_id=5, cv_tag_id=6, setpoint=40.0, kp=2.0, ki=0.8, kd=0.2),
        PIDConfig(pv_tag_id=7, cv_tag_id=8, setpoint=60.0, kp=0.5, ki=0.1, kd=0.01),
    ],
    interlocks=[
        InterlockConfig(
            lolo_limit=0.0,
            low_limit=5.0,
            high_limit=95.0,
            hihi_limit=100.0,
        )
        for _ in range(32)
    ],
)


def build_alarm_engine(config: RoutingConfig) -> AlarmEngine:
    """Builds the tools-core Rust AlarmEngine from the active RoutingConfig."""
    limits_dict = {}
    for i, interlock in enumerate(config.interlocks):
        limits_dict[str(i)] = {
            "lolo": interlock.lolo_limit,
            "low": interlock.low_limit,
            "high": interlock.high_limit,
            "hihi": interlock.hihi_limit,
        }
    return AlarmEngine(limits_dict)


# Global Alarm Engine
alarm_engine = build_alarm_engine(active_config)

# PID Tuning sessions state
tuning_sessions: dict[int, dict[str, Any]] = {}

# Bind references
plc_client.tuning_sessions = tuning_sessions
backup_simulator.tuning_sessions = tuning_sessions
plc_client.active_config = active_config
backup_simulator.active_config = active_config


async def modbus_connect_background() -> None:
    """Periodically attempts to connect to PLC in background without blocking polling loop."""
    logger.info("Starting background PLC connection task...")
    while not shutdown_event.is_set():
        if not plc_client.connected:
            try:
                # Do a non-blocking attempt to connect
                connected = await plc_client.connect()
                if connected:
                    logger.info("Connected to PLC successfully in background.")
            except Exception as e:
                logger.debug(f"Background PLC connect attempt failed: {e}")
        await asyncio.sleep(5.0)


async def poll_plc_loop() -> None:
    """Background loop polling the PLC tags at 10Hz.

    Saves data to DB and streams updates to WS.
    """
    logger.info("Starting background PLC polling loop...")
    while not shutdown_event.is_set():
        try:
            tags = None
            if plc_client.connected:
                tags = await plc_client.read_tags()

            if tags is None:
                # Fallback to simulation step
                tags = await backup_simulator.read_tags()

            # Update latest tags
            if tags is not None:
                for idx, val in enumerate(tags):
                    latest_tags[idx] = val

            # Pack WebSocket message payload containing tags and alicats data
            tag_list = tags if tags is not None else []
            payload = {"tags": tag_list, "alicats": alicat_manager.get_devices_data()}
            await ws_manager.broadcast(payload)

            if tags is not None:
                db_session = None
                try:
                    db_session = next(get_session())
                    # Log tags
                    for tag_id, value in enumerate(tags):
                        log_entry = TagLog(tag_id=tag_id, value=value)
                        db_session.add(log_entry)

                        # Process Alarms
                        events = alarm_engine.update_tag(str(tag_id), value)
                        for ev in events:
                            cur_state = str(ev["current_state"]).split(".")[-1]
                            sev = 0
                            if cur_state in ["Low", "High"]:
                                sev = 1
                            elif cur_state in ["LoLo", "HiHi"]:
                                sev = 2
                            event_log = EventLog(
                                event_type="ALARM",
                                description=f"Tag {tag_id} crossed limit. State: {cur_state} Value: {value}",
                                severity=sev,
                            )
                            db_session.add(event_log)

                    db_session.commit()
                except Exception as db_err:
                    if db_session:
                        db_session.rollback()
                    logger.error(f"Error logging tags/alarms: {db_err}")
                finally:
                    if db_session:
                        db_session.close()

        except Exception as loop_err:
            logger.error(f"Unexpected error in PLC polling loop: {loop_err}")

        # Sleep to maintain 10Hz frequency (100ms cycle)
        await asyncio.sleep(0.1)

    logger.info("Background PLC polling loop stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: initialize database and start PLC polling thread & Alicat manager
    init_db()
    shutdown_event.clear()
    alicat_manager.start()
    connect_task = asyncio.create_task(modbus_connect_background())
    polling_task = asyncio.create_task(poll_plc_loop())
    yield
    # Shutdown: signal task stop, close client connection & Alicat manager
    shutdown_event.set()
    await connect_task
    await polling_task
    await alicat_manager.stop()
    await plc_client.disconnect()


app = FastAPI(
    title="P1AM DCS SCADA SCADA Middleware",
    description="Middleware bridging the P1AM PLC and HMI Dashboard.",
    lifespan=lifespan,
)

# Enable CORS for Vite frontend running on port 3002
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root_info() -> str:
    """HTML landing page directing users to HMI dashboard or API documentation."""
    return """
    <html>
        <head>
            <title>P1AM DCS SCADA Backend</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #0b0d12;
                    color: #ffffff;
                    padding: 3rem;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 80vh;
                    margin: 0;
                }
                .container {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                    padding: 2.5rem;
                    max-width: 600px;
                    text-align: center;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
                }
                h1 {
                    font-size: 1.8rem;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                p {
                    color: #9ea8b6;
                    margin-bottom: 2rem;
                }
                .links {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                a {
                    display: block;
                    padding: 0.85rem;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    color: #00f2fe;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: 600;
                    transition: all 0.2s ease;
                }
                a:hover {
                    background: rgba(0, 242, 254, 0.1);
                    border-color: #00f2fe;
                    transform: translateY(-2px);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>P1AM DCS SCADA Middleware</h1>
                <p>The backend API service is running successfully.</p>
                <div class="links">
                    <a
                        href="http://localhost:3002"
                        target="_blank"
                    >
                        Open HMI Dashboard (Port 3002)
                    </a>
                    <a href="/docs">Open Swagger API Docs (Port 8000)</a>
                </div>
            </div>
        </body>
    </html>
    """


@app.get("/api/routing", response_model=RoutingConfig)
async def get_routing() -> RoutingConfig:
    """Read the active routing and PID parameters from the PLC.

    Returns:
        RoutingConfig: The current routing parameters from the PLC.
    """
    config = await plc_client.read_routing()
    if config is None:
        config = await backup_simulator.read_routing()
    if config is None:
        raise HTTPException(
            status_code=500, detail="Failed to read routing configuration."
        )
    return config


@app.post("/api/routing")
async def update_routing(config: RoutingConfig) -> dict[str, str]:
    """Write new routing configurations to the PLC.

    Args:
        config: RoutingConfig model.

    Returns:
        JSON response indicating success.
    """
    global active_config
    active_config = config
    plc_client.active_config = config
    backup_simulator.active_config = config

    if not plc_client.connected:
        await backup_simulator.write_routing(config)
        return {
            "status": "success",
            "message": "Configuration successfully applied to simulated PLC.",
        }

    success = await plc_client.write_routing(config)
    await backup_simulator.write_routing(config)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to write routing parameters to PLC registers.",
        )

    save_success = await plc_client.save_to_flash()
    await backup_simulator.save_to_flash()

    if not save_success:
        raise HTTPException(
            status_code=500,
            detail=(
                "Config registers written, but failed to trigger 'Save to Flash' coil."
            ),
        )

    return {
        "status": "success",
        "message": ("Configuration successfully deployed and saved to PLC NVRAM."),
    }


@app.post("/api/estop")
async def trigger_estop() -> dict[str, str]:
    """Immediate safety shutdown command, zeroing all tag variables."""
    global latest_tags
    if not plc_client.connected:
        await backup_simulator.trigger_estop()
        latest_tags = [0.0] * 32
        return {
            "status": "success",
            "message": "Simulated E-stop triggered. All simulated tag values zeroed.",
        }

    success = await plc_client.trigger_estop()
    await backup_simulator.trigger_estop()
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to transmit emergency stop command to PLC.",
        )

    return {
        "status": "success",
        "message": ("E-stop triggered successfully. All PLC outputs driven to 0."),
    }


@app.get("/api/export")
async def export_data(
    tag_ids: str = Query(..., description="Comma-separated list of Tag IDs"),
    start_time: str = Query(..., description="Start date ISO string"),
    end_time: str = Query(..., description="End date ISO string"),
    db: Session = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """Exports logged tag historical states as a downloadable CSV file.

    Returns:
        StreamingResponse: Streaming CSV data.
    """
    try:
        parsed_tag_ids = [int(tid.strip()) for tid in tag_ids.split(",") if tid.strip()]
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except ValueError as val_err:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter formats: {val_err}",
        ) from val_err

    # Query time-series logs
    statement = (
        select(TagLog)
        .where(col(TagLog.tag_id).in_(parsed_tag_ids))
        .where(col(TagLog.timestamp) >= start_dt)
        .where(col(TagLog.timestamp) <= end_dt)
        .order_by(col(TagLog.timestamp).asc())
    )
    results = db.exec(statement).all()

    # Generate CSV in memory stream
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Tag ID", "Value"])

    for row in results:
        writer.writerow([row.timestamp.isoformat(), row.tag_id, row.value])

    output.seek(0)
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
    )
    timestamp_sec = int(datetime.now(UTC).timestamp())
    response.headers["Content-Disposition"] = (
        f"attachment; filename=tag_export_{timestamp_sec}.csv"
    )
    return response


class TagWritePayload(BaseModel):
    value: float


@app.post("/api/tags/{tag_id}")
async def write_tag_value(tag_id: int, payload: TagWritePayload) -> dict[str, str]:
    """Manually force/write a 32-bit float value directly to a tag register."""
    if not (0 <= tag_id < 32):
        raise HTTPException(
            status_code=400,
            detail="Tag ID must be between 0 and 31.",
        )

    global latest_tags
    if not plc_client.connected:
        await backup_simulator.write_tag(tag_id, payload.value)
        latest_tags[tag_id] = payload.value
        return {
            "status": "success",
            "message": f"Successfully forced simulated tag {tag_id} to {payload.value}.",
        }

    success = await plc_client.write_tag(tag_id, payload.value)
    await backup_simulator.write_tag(tag_id, payload.value)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write value {payload.value} to tag {tag_id}.",
        )

    latest_tags[tag_id] = payload.value
    return {
        "status": "success",
        "message": f"Successfully wrote {payload.value} to tag {tag_id}.",
    }


@app.post("/api/pid/{pid_index}/tuning/start")
async def start_pid_tuning(pid_index: int) -> dict[str, str]:
    """Decouples the PID loop from automatic control and begins logging step change history."""
    if not (0 <= pid_index < 4):
        raise HTTPException(
            status_code=400, detail="PID index must be between 0 and 3."
        )

    pv_id = active_config.pids[pid_index].pv_tag_id
    cv_id = active_config.pids[pid_index].cv_tag_id
    current_pv = latest_tags[pv_id]
    current_cv = latest_tags[cv_id]

    tuning_sessions[pid_index] = {
        "start_time": time.time(),
        "history": [],
        "step_triggered": False,
        "step_time": 0.0,
        "initial_cv": current_cv,
        "initial_pv": current_pv,
        "final_cv": current_cv,
        "final_pv": current_pv,
    }
    logger.info(f"Started tuning mode for PID loop {pid_index}")
    return {
        "status": "success",
        "message": f"Tuning mode started for PID loop {pid_index}.",
    }


@app.post("/api/pid/{pid_index}/tuning/step")
async def step_pid_tuning(
    pid_index: int, payload: PIDTuningStepPayload
) -> dict[str, str]:
    """Executes a step change in the loop's control variable (CV)."""
    if pid_index not in tuning_sessions:
        raise HTTPException(
            status_code=400, detail="Tuning session not active for this PID loop."
        )

    session = tuning_sessions[pid_index]
    cv_id = active_config.pids[pid_index].cv_tag_id

    session["step_triggered"] = True
    session["step_time"] = time.time() - session["start_time"]
    session["initial_cv"] = latest_tags[cv_id]
    session["final_cv"] = payload.step_value

    await plc_client.write_tag(cv_id, payload.step_value)
    await backup_simulator.write_tag(cv_id, payload.step_value)
    latest_tags[cv_id] = payload.step_value

    logger.info(
        f"Tuning step triggered on loop {pid_index}: CV set to {payload.step_value}"
    )
    return {
        "status": "success",
        "message": f"Step change applied. CV set to {payload.step_value}.",
    }


@app.post("/api/pid/{pid_index}/tuning/stop")
async def stop_pid_tuning(pid_index: int) -> dict[str, Any]:
    """Stops the tuning session, calculates FOPDT process parameters, and recommends tuned gains."""
    if pid_index not in tuning_sessions:
        raise HTTPException(
            status_code=400, detail="Tuning session not active for this PID loop."
        )

    session = tuning_sessions.pop(pid_index)
    history = session["history"]

    if not history or not session["step_triggered"]:
        return {
            "status": "warning",
            "message": "Tuning stopped, but no step change was executed or history is empty.",
            "parameters": {"kp": 0.0, "tau": 0.0, "theta": 0.0},
            "recommended_pid": {"kp": 0.0, "ki": 0.0, "kd": 0.0},
        }

    delta_cv = session["final_cv"] - session["initial_cv"]
    if abs(delta_cv) < 0.01:
        delta_cv = 1.0

    n_samples = len(history)
    last_samples = history[max(0, n_samples - 10) :]
    final_pv = sum(h[2] for h in last_samples) / len(last_samples)
    initial_pv = session["initial_pv"]
    delta_pv = final_pv - initial_pv

    kp_ident = delta_pv / delta_cv

    threshold_10 = initial_pv + 0.10 * delta_pv
    threshold_63 = initial_pv + 0.632 * delta_pv

    t_step = session["step_time"]
    t_10 = None
    t_63 = None

    for time_offset, _, pv_val in history:
        if time_offset < t_step:
            continue
        if t_10 is None:
            if (delta_pv > 0 and pv_val >= threshold_10) or (
                delta_pv < 0 and pv_val <= threshold_10
            ):
                t_10 = time_offset
        if t_63 is None:
            if (delta_pv > 0 and pv_val >= threshold_63) or (
                delta_pv < 0 and pv_val <= threshold_63
            ):
                t_63 = time_offset

    if t_10 is None:
        t_10 = t_step + 1.0
    if t_63 is None:
        t_63 = t_10 + 2.0

    theta_ident = max(0.1, t_10 - t_step)
    tau_ident = max(0.1, t_63 - t_10)

    ratio = theta_ident / tau_ident
    if abs(kp_ident) > 0.001:
        kc = (1.0 / kp_ident) * (tau_ident / theta_ident) * (1.333 + 0.25 * ratio)
        ti = theta_ident * (32.0 + 6.0 * ratio) / (13.0 + 8.0 * ratio)
        td = theta_ident * 4.0 / (11.0 + 2.0 * ratio)

        kp_recom = round(kc, 3)
        ki_recom = round(kc / ti, 3)
        kd_recom = round(kc * td, 3)
    else:
        kp_recom = 0.0
        ki_recom = 0.0
        kd_recom = 0.0

    return {
        "status": "success",
        "message": "Tuning parameters identified successfully.",
        "parameters": {
            "kp": round(kp_ident, 3),
            "tau": round(tau_ident, 2),
            "theta": round(theta_ident, 2),
        },
        "recommended_pid": {
            "kp": max(0.0, kp_recom),
            "ki": max(0.0, ki_recom),
            "kd": max(0.0, kd_recom),
        },
    }


class MPCSimulatePayload(BaseModel):
    prediction_horizon: int = PydanticField(10, ge=2, le=30)
    control_horizon: int = PydanticField(3, ge=1, le=10)
    setpoint: float = PydanticField(50.0, ge=0.0, le=100.0)
    rho: float = PydanticField(0.1, ge=0.0, le=10.0)
    process_gain: float = PydanticField(1.2, ge=0.1, le=5.0)
    process_tau: float = PydanticField(5.0, ge=0.5, le=20.0)
    process_delay: float = PydanticField(1.0, ge=0.0, le=5.0)


@app.post("/api/mpc/simulate")
async def simulate_mpc(payload: MPCSimulatePayload) -> dict[str, Any]:
    """Simulates and compares standard PID versus Model Predictive Control (MPC)."""
    Kp = payload.process_gain
    tau = payload.process_tau
    theta = payload.process_delay
    dt = 0.5
    steps = 50

    # PID Simulation
    ratio = max(0.1, theta) / max(0.5, tau)
    kc = (1.0 / Kp) * (tau / max(0.1, theta)) * (1.333 + 0.25 * ratio)
    ti = max(0.1, theta) * (32.0 + 6.0 * ratio) / (13.0 + 8.0 * ratio)
    td = max(0.1, theta) * 4.0 / (11.0 + 2.0 * ratio)
    pid_kp = kc
    pid_ki = kc / ti
    pid_kd = kc * td

    pid_pv = [0.0] * steps
    pid_cv = [0.0] * steps
    pid_integral = 0.0
    pid_prev_err = 0.0
    cv_hist_pid = [0.0] * steps

    for k in range(1, steps):
        err = payload.setpoint - pid_pv[k - 1]
        pid_integral = max(-100.0, min(100.0, pid_integral + err * dt))
        deriv = (err - pid_prev_err) / dt
        pid_prev_err = err

        cv = pid_kp * err + pid_ki * pid_integral + pid_kd * deriv
        cv = max(0.0, min(100.0, cv))
        pid_cv[k] = cv
        cv_hist_pid[k] = cv

        delay_idx = k - int(theta / dt)
        delayed_cv = cv_hist_pid[max(0, delay_idx)]
        dy = (Kp * delayed_cv - pid_pv[k - 1]) * (dt / tau)
        pid_pv[k] = max(0.0, pid_pv[k - 1] + dy)

    # MPC Simulation using Projected Gradient Descent
    mpc_pv = [0.0] * steps
    mpc_cv = [0.0] * steps
    cv_hist_mpc = [0.0] * steps

    P = payload.prediction_horizon
    M = payload.control_horizon

    for k in range(1, steps):
        g = []
        for j in range(1, P + 1):
            t_eval = j * dt - theta
            if t_eval <= 0:
                g.append(0.0)
            else:
                g.append(Kp * (1.0 - math.exp(-t_eval / tau)))

        G = [[0.0] * M for _ in range(P)]
        for r in range(P):
            for c in range(M):
                if r >= c:
                    G[r][c] = g[r - c]

        f = []
        last_u = mpc_cv[k - 1]
        for j in range(1, P + 1):
            t_eval = j * dt - theta
            if t_eval <= 0:
                f.append(mpc_pv[k - 1])
            else:
                f.append(
                    mpc_pv[k - 1]
                    + (Kp * last_u - mpc_pv[k - 1]) * (1.0 - math.exp(-t_eval / tau))
                )

        r_vec = [payload.setpoint] * P

        GTG = [[0.0] * M for _ in range(M)]
        for r_idx in range(M):
            for c_idx in range(M):
                val = 0.0
                for p_idx in range(P):
                    val += G[p_idx][r_idx] * G[p_idx][c_idx]
                GTG[r_idx][c_idx] = val

        H = [[2.0 * GTG[r_idx][c_idx] for c_idx in range(M)] for r_idx in range(M)]
        for i in range(M):
            H[i][i] += 2.0 * payload.rho

        c_vec = [0.0] * M
        for c_idx in range(M):
            val = 0.0
            for p_idx in range(P):
                val += G[p_idx][c_idx] * (f[p_idx] - r_vec[p_idx])
            c_vec[c_idx] = 2.0 * val

        u_opt = [last_u] * M
        alpha = 0.01 / (
            2.0 * (sum(sum(abs(x) for x in row) for row in GTG) + payload.rho + 1.0)
        )

        for _ in range(100):
            grad = [0.0] * M
            for r_idx in range(M):
                val = 0.0
                for c_idx in range(M):
                    val += H[r_idx][c_idx] * u_opt[c_idx]
                grad[r_idx] = val + c_vec[r_idx]

            for i in range(M):
                u_opt[i] = max(0.0, min(100.0, u_opt[i] - alpha * grad[i]))

        mpc_cv[k] = u_opt[0]
        cv_hist_mpc[k] = mpc_cv[k]

        delay_idx = k - int(theta / dt)
        delayed_cv = cv_hist_mpc[max(0, delay_idx)]
        dy = (Kp * delayed_cv - mpc_pv[k - 1]) * (dt / tau)
        mpc_pv[k] = max(0.0, mpc_pv[k - 1] + dy)

    time_series = [round(i * dt, 1) for i in range(steps)]
    return {
        "status": "success",
        "time": time_series,
        "pid": {
            "pv": [round(x, 2) for x in pid_pv],
            "cv": [round(x, 2) for x in pid_cv],
        },
        "mpc": {
            "pv": [round(x, 2) for x in mpc_pv],
            "cv": [round(x, 2) for x in mpc_cv],
        },
    }


@app.get("/api/alicats", response_model=list[AlicatMFCState])
async def get_alicats() -> list[AlicatMFCState]:
    """Retrieve live parameters for all configured Alicat Mass Flow Controllers."""
    return [AlicatMFCState(**d) for d in alicat_manager.get_devices_data()]


@app.post("/api/alicats/{device_id}/setpoint")
async def update_alicat_setpoint(
    device_id: str, payload: AlicatSetpointPayload
) -> dict[str, str]:
    """Modify the flow setpoint for a specific mass flow controller."""
    success = alicat_manager.update_mfc_setpoint(device_id, payload.setpoint)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alicat MFC '{device_id}' not found.",
        )
    return {
        "status": "success",
        "message": f"Setpoint for MFC '{device_id}' set to {payload.setpoint} SLPM.",
    }


@app.post("/api/alicats/{device_id}/gas")
async def update_alicat_gas(
    device_id: str, payload: AlicatGasPayload
) -> dict[str, str]:
    """Modify the active gas calibration for a specific mass flow controller."""
    success = alicat_manager.update_mfc_gas(device_id, payload.gas)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alicat MFC '{device_id}' not found or gas type invalid.",
        )
    return {
        "status": "success",
        "message": f"Gas species for MFC '{device_id}' set to {payload.gas}.",
    }


@app.websocket("/api/stream")
async def stream_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint streaming live PLC Tag values to the client HMI."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # We must continuously receive messages (even ping/pong) to keep socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection exception: {e}")
        ws_manager.disconnect(websocket)


@app.get("/api/alarms/active")
async def get_active_alarms() -> list[dict[str, Any]]:
    """Retrieve all active, unacknowledged alarms."""
    active = []
    for tag_id, state in alarm_engine.tag_states.items():
        state_str = str(state).split(".")[-1]
        if state_str != "Normal" and not alarm_engine.tag_acknowledged.get(
            tag_id, False
        ):
            active.append(
                {
                    "tag_id": tag_id,
                    "state": state_str,
                    "value": alarm_engine.tag_values.get(tag_id, 0.0),
                }
            )
    return active


@app.post("/api/alarms/{tag_id}/acknowledge")
async def acknowledge_alarm(tag_id: str) -> dict[str, str]:
    """Acknowledge an active alarm."""
    try:
        success = alarm_engine.acknowledge_alarm(tag_id, "Operator")

        # Log to db
        db_session = next(get_session())
        log_entry = EventLog(
            event_type="ACKNOWLEDGE",
            description=f"Alarm on Tag {tag_id} acknowledged by Operator.",
            severity=0,
        )
        db_session.add(log_entry)
        db_session.commit()
        db_session.close()

        if success:
            return {
                "status": "success",
                "message": f"Alarm on tag {tag_id} acknowledged.",
            }
        return {"status": "ignored", "message": "No unacknowledged alarm for this tag."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/events")
async def get_events(
    limit: int = 100,
    db: Session = Depends(get_session),  # noqa: B008
) -> list[EventLog]:
    """Retrieve event history."""
    stmt = select(EventLog).order_by(col(EventLog.timestamp).desc()).limit(limit)
    return list(db.exec(stmt).all())
