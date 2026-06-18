import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
from typing import Any, cast

import historian
from alarm_processing import process_alarm_events
from alicat_manager import AlicatManager, AlicatMFC
from auth_config import require_admin_key, require_api_key, verify_operator_key
from cors_config import resolve_cors_settings
from data_capture import (
    TRENDS_MAX_POINTS,
    CaptureStats,
    ClearResult,
    capture_stats,
    clear_capture,
    historian_retention_loop,
    parse_query_bound,
    parse_tag_names,
    stream_tag_export_csv,
)
from database import engine, get_session, init_db
from defaults import default_routing_config
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
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
    PIDTuningStepPayload,
    PlantArea,
    PlantEquipment,
    PlantUnit,
    RoutingConfig,
    TagDefinitionDb,
    TagLog,
)
from plant_model import TagDefinition
from plc_factory import PLCFactory
from power_supply_integration import PowerSupplyService, create_power_supply_router
from power_supply_passthrough import ensure_power_supply_passthrough
from project_import import import_project_archive
from pydantic import BaseModel
from pydantic import Field as PydanticField
from simulator_client import SimulatedPLCClient
from sqlmodel import Session, col, select

try:
    # Prefer the Rust-accelerated SCADA kernel when the compiled wheel is
    # installed. It is shipped as the PyO3 ``tools_core`` extension and is not
    # present in every environment (fresh checkout, no Rust toolchain, slim
    # deployment image), so guard the import and fall back to the pure-Python
    # implementation rather than failing the whole backend at import time.
    from tools_core import scada
except ModuleNotFoundError:
    import scada_fallback as scada

    logging.getLogger("dcs_backend.main").warning(
        "tools_core wheel not installed; using pure-Python scada fallback. "
        "Build the Rust extension for accelerated SCADA performance "
        "(maturin build --release --features python,extension-module "
        "-m rust_core/tools-core/Cargo.toml)."
    )

AlarmEngine = scada.AlarmEngine
exponential_smoothing = scada.exponential_smoothing
moving_average = scada.moving_average

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dcs_backend.main")

plc_client = PLCFactory.create_client()
modbus_manager = plc_client  # Compatibility alias
backup_simulator = SimulatedPLCClient()

power_supply_service = PowerSupplyService(plc_client=plc_client, logger=logger)


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
        # Iterate a snapshot and prune any client whose send fails — otherwise a
        # dead socket lingers forever and gets re-tried (and re-logged) at 10 Hz,
        # a slow leak that degrades the loop over a long session.
        dead: list[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Dropping unreachable WebSocket client: {e}")
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


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

latest_tags: dict[str, float] = {f"TAG_{i}": 0.0 for i in range(32)}
active_config: RoutingConfig = default_routing_config()


def load_tags_into_plc_clients(session: Session) -> None:
    """Loads all tag definitions from the database and registers them with the PLC clients."""
    tag_defs = session.exec(select(TagDefinitionDb)).all()
    if not tag_defs:
        return

    # Build the tag definition mapping
    tag_map = {}
    for td in tag_defs:
        tag_map[td.name] = TagDefinition(
            name=td.name,
            tag_type=td.tag_type,
            description=td.description,
            rw_mode=td.rw_mode,
            register_type=td.register_type,
            register_num=td.register_num,
            data_format=td.data_format,
            scale_factor=td.scale_factor,
        )

    # Set them on clients
    plc_client.tag_map = tag_map
    backup_simulator.tag_map = tag_map


def build_alarm_engine(config: RoutingConfig) -> Any:
    """Builds the tools-core Rust AlarmEngine from the active RoutingConfig."""
    limits_dict = {}
    for tag_name, interlock in config.interlocks.items():
        limits_dict[tag_name] = {
            "lolo": interlock.lolo_limit,
            "low": interlock.low_limit,
            "high": interlock.high_limit,
            "hihi": interlock.hihi_limit,
        }
    return AlarmEngine(limits_dict)


alarm_engine = build_alarm_engine(active_config)

active_alarms: dict[str, dict[str, Any]] = {}
e_stop_active: bool = False


def apply_alarm_config(config: RoutingConfig) -> None:
    """Rebuild the alarm engine from `config` and clear stale active alarms.

    Keeps the alarm set in sync with the active interlock configuration. Called
    on PLC connect (to adopt the device's real interlock limits instead of the
    startup defaults — otherwise every tag resting at 0 trips the default
    LoLo/Low limits) and whenever routing is updated.
    """
    global alarm_engine
    alarm_engine = build_alarm_engine(config)
    active_alarms.clear()


tuning_sessions: dict[int, dict[str, Any]] = {}

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
                    if e_stop_active:
                        try:
                            await plc_client.trigger_estop()
                            logger.warning("Re-asserted hardware E-stop on reconnect.")
                        except Exception as estop_err:
                            logger.error(f"Failed to re-assert E-stop: {estop_err}")
                    # Adopt the device's real routing + interlock limits so the
                    # alarm set reflects the PLC, not the startup defaults.
                    try:
                        plc_config = await plc_client.read_routing()
                        if plc_config is not None:
                            # If the power-supply PID came up unmapped after an
                            # NVRAM reset, auto-repair it to a pass-through so a
                            # commanded setpoint actually drives the AO (#3550).
                            plc_config = await ensure_power_supply_passthrough(
                                plc_client,
                                plc_config,
                                command_tag=power_supply_service.controller.config.command_tag,
                                logger=logger,
                            )
                            global active_config
                            active_config = plc_config
                            apply_alarm_config(plc_config)
                            logger.info("Synced routing and alarm limits from PLC.")
                    except Exception as sync_err:
                        logger.warning(f"Could not sync routing from PLC: {sync_err}")
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
                for key, val in tags.items():
                    latest_tags[key] = val
            # Pack WebSocket message payload containing tags and alicats data
            tag_list = (
                [tags.get(f"TAG_{i}", 0.0) for i in range(32)]
                if tags is not None
                else []
            )
            # The controller is latched while e_stop_active, so poll() commands
            # zero. Re-assert the hardware kill every scan as well, so a stale
            # PID setpoint or any other driver can never re-energize the loop
            # while the E-stop is held.
            ps_status = await power_supply_service.poll(tags)
            if e_stop_active and plc_client.connected:
                await plc_client.trigger_estop()

            payload = {
                "tags": tag_list,
                "tags_dict": tags if tags is not None else {},
                "alicats": alicat_manager.get_devices_data(),
                "active_alarms": active_alarms,
                "e_stop_active": e_stop_active,
                "power_supply": ps_status.model_dump(),
            }
            await ws_manager.broadcast(payload)
            if tags is not None:
                db_session = None
                try:
                    db_session = next(get_session())
                    # Bulk-insert this scan's samples (cheap single INSERT), then
                    # fold alarm transitions into active_alarms and persist their
                    # event rows — all under one commit.
                    historian.log_scan(db_session, tags)
                    for event_log in process_alarm_events(
                        alarm_engine, tags, active_alarms
                    ):
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
    with Session(engine) as session:
        load_tags_into_plc_clients(session)
    shutdown_event.clear()
    alicat_manager.start()
    connect_task = asyncio.create_task(modbus_connect_background())
    polling_task = asyncio.create_task(poll_plc_loop())
    retention_task = asyncio.create_task(
        historian_retention_loop(
            shutdown_event=shutdown_event,
            engine=engine,
            logger=logger,
        )
    )
    yield
    # Shutdown: signal task stop, close client connection & Alicat manager
    shutdown_event.set()
    await connect_task
    await polling_task
    await retention_task
    await alicat_manager.stop()
    await plc_client.disconnect()


app = FastAPI(
    title="P1AM DCS SCADA SCADA Middleware",
    description="Middleware bridging the P1AM PLC and HMI Dashboard.",
    lifespan=lifespan,
)
app.include_router(create_power_supply_router(power_supply_service))

# Restrict CORS to a configured allowlist (no wildcard with credentials).
# See cors_config.resolve_cors_settings for env-driven configuration.
_cors = resolve_cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_cors.allow_origins),
    allow_credentials=_cors.allow_credentials,
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


@app.post("/api/routing", dependencies=[Depends(require_admin_key)])
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
    # Keep alarms in sync with the new interlock limits.
    apply_alarm_config(config)

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


# NOTE: E-stop *activation* is intentionally left unauthenticated so a panic
# stop is always reachable even without a credential (safety over confidentiality).
# Clearing the E-stop (below) requires the admin credential.
@app.post("/api/estop")
async def trigger_estop() -> dict[str, str]:
    """Immediate safety shutdown command, zeroing all tag variables."""
    global latest_tags, e_stop_active
    e_stop_active = True
    # Latch the controller FIRST so the next poll cycle cannot re-command a
    # setpoint and re-energize the output after we zero it below.
    power_supply_service.engage_estop()
    if not plc_client.connected:
        await backup_simulator.trigger_estop()
        latest_tags = {f"TAG_{i}": 0.0 for i in range(32)}
        return {
            "status": "success",
            "message": "Simulated E-stop triggered. All simulated tag values zeroed.",
        }

    ok = await plc_client.trigger_estop()
    await backup_simulator.trigger_estop()
    if not ok:
        raise HTTPException(
            status_code=502,
            detail="E-stop command was not acknowledged by the PLC; controller remains latched and will retry.",
        )
    return {"status": "success", "message": "Hardware E-stop triggered."}


@app.post("/api/estop/clear", dependencies=[Depends(require_admin_key)])
async def clear_estop() -> dict[str, str]:
    """Clear the E-stop state on the controller, then in the HMI.

    The E-stop latch lives in the PLC, not in this process. We MUST command the
    controller to reset before reporting the HMI as cleared; otherwise the header
    turns green while the plant is still tripped. The server-side ``e_stop_active``
    flag is only lowered once the controller (or the backup simulator, when the
    PLC is offline) acknowledges the reset.
    """
    global e_stop_active

    if not plc_client.connected:
        cleared = await backup_simulator.clear_estop()
        if not cleared:
            raise HTTPException(
                status_code=502,
                detail="Backup simulator did not acknowledge the E-stop reset.",
            )
        e_stop_active = False
        return {
            "status": "success",
            "message": "Simulated E-stop cleared.",
        }

    cleared = await plc_client.clear_estop()
    if not cleared:
        # Leave e_stop_active latched so the HMI keeps showing the tripped state.
        raise HTTPException(
            status_code=502,
            detail=("PLC did not acknowledge the E-stop reset; plant remains tripped."),
        )
    await backup_simulator.clear_estop()
    e_stop_active = False
    # Release the power-supply controller latch too. It returns to IDLE with
    # permissive off, so the operator must deliberately re-arm before any
    # output can flow.
    power_supply_service.clear_estop()
    return {"status": "success", "message": "Hardware E-stop cleared."}


@app.get("/api/alarms/active")
async def get_active_alarms() -> list[dict[str, Any]]:
    """Get all currently active or unacknowledged alarms."""
    return list(active_alarms.values())


@app.post(
    "/api/alarms/{tag_id}/acknowledge",
    dependencies=[Depends(require_api_key)],
)
async def acknowledge_alarm(
    tag_id: str,
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, str]:
    """Acknowledge a specific active alarm."""
    if tag_id in active_alarms:
        active_alarms[tag_id]["acknowledged"] = True

        # Log the acknowledgment
        try:
            event_log = EventLog(
                event_type="ACKNOWLEDGE",
                description=f"Alarm on Tag {tag_id} acknowledged by user.",
                severity=0,
            )
            db.add(event_log)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log acknowledgment: {e}")
            db.rollback()

        # If it returned to normal already, we can remove it now
        if active_alarms[tag_id]["state"] == "Normal":
            del active_alarms[tag_id]

        return {"status": "success", "message": f"Alarm {tag_id} acknowledged."}
    return {"status": "ignored", "message": f"Alarm {tag_id} not found."}


class EventLogPayload(BaseModel):
    event_type: str
    description: str


@app.post("/api/events", dependencies=[Depends(require_api_key)])
async def log_user_event(
    payload: EventLogPayload,
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, str]:
    """Log a user action or system event."""
    try:
        event_log = EventLog(
            event_type=payload.event_type,
            description=payload.description,
            severity=0,
        )
        db.add(event_log)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to log user event: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to log event.") from e


@app.get("/api/events")
async def get_events(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_session),  # noqa: B008
) -> list[EventLog]:
    """Fetch paginated historical event logs."""
    statement = (
        select(EventLog)
        .order_by(col(EventLog.timestamp).desc())
        .offset(offset)
        .limit(limit)
    )
    results = db.exec(statement).all()
    return list(results)


@app.get("/api/trends")
async def get_trends(
    tag_id: str,
    start_time: str,
    end_time: str,
    smoothing: str = "none",
    window_size: int = 5,
    alpha: float = 0.2,
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, Any]:
    """Fetch bounded historical trends, optionally applying server-side smoothing."""
    try:
        start_dt = parse_query_bound(start_time)
        end_dt = parse_query_bound(end_time)
    except (TypeError, ValueError) as val_err:
        raise HTTPException(
            status_code=400, detail=f"Invalid date format: {val_err}"
        ) from val_err

    tag_name = tag_id
    if tag_id.isdigit():
        tag_name = f"TAG_{tag_id}"

    # Cap the row count: take the most-recent N within the range (DESC + limit),
    # then present oldest-first.
    statement = (
        select(TagLog)
        .where(col(TagLog.tag_name) == tag_name)
        .where(col(TagLog.timestamp) >= start_dt)
        .where(col(TagLog.timestamp) <= end_dt)
        .order_by(col(TagLog.timestamp).desc())
        .limit(TRENDS_MAX_POINTS)
    )
    results = list(reversed(db.exec(statement).all()))
    truncated = len(results) >= TRENDS_MAX_POINTS

    timestamps = [r.timestamp.isoformat() for r in results]
    values = [float(r.value) for r in results]

    if smoothing == "moving_average" and values:
        values = moving_average(values, window_size)
    elif smoothing == "exponential_smoothing" and values:
        values = exponential_smoothing(values, alpha)

    return {"timestamps": timestamps, "values": values, "truncated": truncated}


@app.get("/api/export")
async def export_data(
    tag_ids: str = Query(..., description="Comma-separated list of Tag IDs or Names"),
    start_time: str = Query(..., description="Start date ISO string"),
    end_time: str = Query(..., description="End date ISO string"),
    db: Session = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """Exports logged tag historical states as a downloadable CSV file.

    Returns:
        StreamingResponse: Streaming CSV data.
    """
    try:
        parsed_tag_names = parse_tag_names(tag_ids)
        start_dt = parse_query_bound(start_time)
        end_dt = parse_query_bound(end_time)
    except (TypeError, ValueError) as val_err:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameter formats: {val_err}",
        ) from val_err

    statement = (
        select(TagLog)
        .where(col(TagLog.tag_name).in_(parsed_tag_names))
        .where(col(TagLog.timestamp) >= start_dt)
        .where(col(TagLog.timestamp) <= end_dt)
        .order_by(col(TagLog.timestamp).asc())
    )

    bind = db.get_bind()

    timestamp_sec = int(datetime.now(UTC).timestamp())
    return StreamingResponse(
        stream_tag_export_csv(bind, statement),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=tag_export_{timestamp_sec}.csv"
        },
    )


@app.get("/api/capture/status", response_model=CaptureStats)
async def get_capture_status(
    db: Session = Depends(get_session),  # noqa: B008
) -> CaptureStats:
    """Report the captured historian: rows, time span, distinct tags, disk size.

    Capture is automatic — the polling loop logs every scan whenever the backend
    is up — so ``capturing`` mirrors that always-on behavior for the HMI's REC
    indicator.
    """
    return capture_stats(db, capturing=True)


class CaptureClearRequest(BaseModel):
    """Operator request to clear captured data."""

    include_events: bool = False


@app.post(
    "/api/capture/clear",
    response_model=ClearResult,
    dependencies=[Depends(require_admin_key)],
)
async def clear_capture_data(
    req: CaptureClearRequest,
    db: Session = Depends(get_session),  # noqa: B008
) -> ClearResult:
    """Clear the captured historian and reclaim disk (VACUUM).

    A destructive maintenance action — admin-gated — so a long test campaign
    cannot silently fill the storage device. Optionally clears the event log too.
    """
    result = clear_capture(db, include_events=req.include_events)
    logger.warning(
        "Historian cleared: %d tag rows, %d event rows, %d -> %d bytes",
        result.tag_rows_deleted,
        result.event_rows_deleted,
        result.db_bytes_before,
        result.db_bytes_after,
    )
    return result


class TagWritePayload(BaseModel):
    value: float


@app.post("/api/tags/{tag_id}", dependencies=[Depends(require_admin_key)])
async def write_tag_value(tag_id: str, payload: TagWritePayload) -> dict[str, str]:
    """Manually force/write a 32-bit float value directly to a tag register."""
    global latest_tags
    tag_name = tag_id
    if tag_id.isdigit():
        val_id = int(tag_id)
        if not (0 <= val_id < 32):
            raise HTTPException(
                status_code=400,
                detail="Tag ID must be between 0 and 31.",
            )
        tag_name = f"TAG_{tag_id}"

    if not plc_client.connected:
        success = await backup_simulator.write_tag(tag_name, payload.value)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Tag '{tag_name}' not found in simulator registry.",
            )
        latest_tags[tag_name] = payload.value
        return {
            "status": "success",
            "message": f"Successfully forced simulated tag {tag_name} to {payload.value}.",
        }

    success = await plc_client.write_tag(tag_name, payload.value)
    await backup_simulator.write_tag(tag_name, payload.value)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to write value {payload.value} to tag {tag_name}.",
        )

    latest_tags[tag_name] = payload.value
    return {
        "status": "success",
        "message": f"Successfully wrote {payload.value} to tag {tag_name}.",
    }


@app.post(
    "/api/pid/{pid_index}/tuning/start",
    dependencies=[Depends(require_admin_key)],
)
async def start_pid_tuning(pid_index: int) -> dict[str, str]:
    """Decouples the PID loop from automatic control and begins logging step change history."""
    if not (0 <= pid_index < 4):
        raise HTTPException(
            status_code=400, detail="PID index must be between 0 and 3."
        )

    pv_tag = active_config.pids[pid_index].pv_tag
    cv_tag = active_config.pids[pid_index].cv_tag
    current_pv = latest_tags[pv_tag]
    current_cv = latest_tags[cv_tag]

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


@app.post(
    "/api/pid/{pid_index}/tuning/step",
    dependencies=[Depends(require_admin_key)],
)
async def step_pid_tuning(
    pid_index: int, payload: PIDTuningStepPayload
) -> dict[str, str]:
    """Executes a step change in the loop's control variable (CV)."""
    if pid_index not in tuning_sessions:
        raise HTTPException(
            status_code=400, detail="Tuning session not active for this PID loop."
        )

    session = tuning_sessions[pid_index]
    cv_tag = active_config.pids[pid_index].cv_tag

    session["step_triggered"] = True
    session["step_time"] = time.time() - session["start_time"]
    session["initial_cv"] = latest_tags[cv_tag]
    session["final_cv"] = payload.step_value

    await plc_client.write_tag(cv_tag, payload.step_value)
    await backup_simulator.write_tag(cv_tag, payload.step_value)
    latest_tags[cv_tag] = payload.step_value

    logger.info(
        f"Tuning step triggered on loop {pid_index}: CV set to {payload.step_value}"
    )
    return {
        "status": "success",
        "message": f"Step change applied. CV set to {payload.step_value}.",
    }


@app.post(
    "/api/pid/{pid_index}/tuning/stop",
    dependencies=[Depends(require_admin_key)],
)
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


@app.post("/api/mpc/simulate", dependencies=[Depends(require_admin_key)])
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


@app.post(
    "/api/alicats/{device_id}/setpoint",
    dependencies=[Depends(require_admin_key)],
)
async def update_alicat_setpoint(
    device_id: str, payload: AlicatSetpointPayload
) -> dict[str, str]:
    """Modify the flow setpoint for a specific mass flow controller."""
    mfc = alicat_manager.devices.get(device_id)
    if mfc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Alicat MFC '{device_id}' not found.",
        )

    success = alicat_manager.update_mfc_setpoint(device_id, payload.setpoint)
    if not success:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Alicat MFC '{device_id}' {mfc.connection_type} physical IO "
                "is unsupported; setpoint was not applied."
            ),
        )
    return {
        "status": "success",
        "message": f"Setpoint for MFC '{device_id}' set to {payload.setpoint} SLPM.",
    }


@app.post("/api/alicats/{device_id}/gas", dependencies=[Depends(require_admin_key)])
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
    """WebSocket endpoint streaming live PLC Tag values to the client HMI.

    Requires a valid operator/admin API key supplied either as the ``api_key``
    query parameter or as the first text frame after connect. Rejects
    unauthenticated connections with policy-violation close code 1008.
    """
    # Validate the credential before accepting the stream (issue #3289).
    provided = websocket.query_params.get("api_key")
    if not verify_operator_key(provided):
        await websocket.accept()
        try:
            first = await websocket.receive_text()
        except Exception:
            await websocket.close(code=1008)
            return
        if not verify_operator_key(first):
            await websocket.close(code=1008)
            return
        # Authenticated via first frame; register without re-accepting.
        ws_manager.active_connections.append(websocket)
        logger.info("New WebSocket client connected (authenticated via frame).")
    else:
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


@app.post("/api/project/import", dependencies=[Depends(require_admin_key)])
async def import_project(
    file: UploadFile = File(...),  # noqa: B008
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, Any]:
    """Upload and ingest a zip file containing tagl.json and PLC driver mapping (.SDV files)."""
    return cast(
        "dict[str, Any]",
        await import_project_archive(file, db, load_tags_into_plc_clients),
    )


@app.get("/api/project/ladder-explorer")
async def get_ladder_explorer(
    db: Session = Depends(get_session),  # noqa: B008
) -> list[dict[str, Any]]:
    """Retrieve all tag definitions with their PLC register mappings for exploration."""
    tags = db.exec(select(TagDefinitionDb)).all()
    results = []
    for t in tags:
        # Load parent equipment, unit, area names
        equip = db.get(PlantEquipment, t.equipment_id) if t.equipment_id else None
        unit = db.get(PlantUnit, equip.unit_id) if equip else None
        area = db.get(PlantArea, unit.area_id) if unit else None

        results.append(
            {
                "name": t.name,
                "tag_type": t.tag_type,
                "description": t.description,
                "rw_mode": t.rw_mode,
                "register_type": t.register_type,
                "register_num": t.register_num,
                "data_format": t.data_format,
                "scale_factor": t.scale_factor,
                "equipment": equip.name if equip else "",
                "unit": unit.name if unit else "",
                "area": area.name if area else "",
            }
        )
    return results


@app.get("/api/plant")
async def get_plant_hierarchy_api(
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, Any]:
    """Retrieve the physical plant layout and tag tree hierarchical structure."""
    areas = db.exec(select(PlantArea)).all()
    units = db.exec(select(PlantUnit)).all()
    equips = db.exec(select(PlantEquipment)).all()
    tags = db.exec(select(TagDefinitionDb)).all()

    # Build tree
    tree: dict[str, Any] = {"name": "Plant", "areas": {}}

    # Index elements
    area_dict = {a.id: {"name": a.name, "units": {}} for a in areas}
    unit_dict = {
        u.id: {"name": u.name, "equipment": {}, "area_id": u.area_id} for u in units
    }
    equip_dict = {
        e.id: {"name": e.name, "tags": {}, "unit_id": e.unit_id} for e in equips
    }

    # Map tags to equipment
    for t in tags:
        if t.equipment_id in equip_dict:
            equip_dict[t.equipment_id]["tags"][t.name] = {
                "name": t.name,
                "tag_type": t.tag_type,
                "description": t.description,
                "rw_mode": t.rw_mode,
                "register_type": t.register_type,
                "register_num": t.register_num,
                "data_format": t.data_format,
                "scale_factor": t.scale_factor,
            }

    # Map equipment to units
    for eq_data in equip_dict.values():
        u_id = eq_data["unit_id"]
        if u_id in unit_dict:
            unit_dict[u_id]["equipment"][eq_data["name"]] = {
                "name": eq_data["name"],
                "tags": eq_data["tags"],
            }

    # Map units to areas
    for u_data in unit_dict.values():
        a_id = u_data["area_id"]
        if a_id in area_dict:
            area_dict[a_id]["units"][u_data["name"]] = {
                "name": u_data["name"],
                "equipment": u_data["equipment"],
            }

    # Map areas to root
    for a_data in area_dict.values():
        tree["areas"][a_data["name"]] = {
            "name": a_data["name"],
            "units": a_data["units"],
        }

    return tree
