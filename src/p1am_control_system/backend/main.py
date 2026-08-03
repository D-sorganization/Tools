import asyncio
import logging
import os
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
from alarm_router import create_alarm_router
from alarm_service import AlarmService, manager_from_routing
from alicat_manager import AlicatManager, AlicatMFC
from audit_middleware import MutationAuditMiddleware
from audit_router import create_audit_router
from auth_config import (
    CREDENTIAL_HEADER_NAME,
    identity_service,
    require_admin_key,
    require_api_key,
    require_engineer_key,
    resolve_optional_principal,
    verify_operator_key,
)
from config_store import load_config, load_model, save_config
from configuration_repository import SqliteRevisionRepository
from configuration_router import create_configuration_router
from configuration_workflow import ConfigurationWorkflow
from cors_config import resolve_cors_settings
from data_capture import (
    TRENDS_MAX_POINTS,
    CaptureConfig,
    CaptureStats,
    CaptureThrottle,
    ClearResult,
    capture_stats,
    clear_capture,
    historian_retention_loop,
    parse_query_bound,
    parse_tag_names,
    query_trend_series,
    stream_tag_export_csv,
)
from database import engine, get_session, init_db
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from identity import Principal
from identity_router import create_identity_router
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
from mpc import simulate_pid_vs_mpc
from performance import PerformanceConfig, PerformanceController, PerformanceMode
from pid_tuning import identify_fopdt_and_tune
from plant_model import TagDefinition
from plc_factory import PLCFactory
from poll_runtime import _connect_once, _poll_once
from power_supply_integration import PowerSupplyService, create_power_supply_router
from project_import import import_project_archive
from pydantic import BaseModel
from pydantic import Field as PydanticField
from settings import get_settings
from signal_quality import SignalFrame
from simulator_client import SimulatedPLCClient
from sqlmodel import Session, col, select
from state import SystemState
from temperature_integration import (
    TemperatureService,
    create_temperature_router,
)

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
settings = get_settings()

# Truthy tokens for the opt-in read-auth env gate (mirrors auth_config).
_TRUTHY = {"1", "true", "yes", "on"}
# auto_error=False so require_read_auth can no-op when the gate is disabled
# instead of FastAPI rejecting a missing header up front.
_read_api_key_header = APIKeyHeader(name=CREDENTIAL_HEADER_NAME, auto_error=False)

plc_client = PLCFactory.create_client(settings)
modbus_manager = plc_client  # Compatibility alias
backup_simulator = SimulatedPLCClient()


# A session factory lets the services durably persist operator settings (config
# + last setpoint) to the config store so they survive a restart. Settings only
# — a restored controller stays IDLE; the operator presses Start to resume.
def _config_session() -> Session:
    return Session(engine)


def _persist_setting(key: str, payload: dict[str, object]) -> None:
    """Best-effort persist of a small operator setting to the config store.

    Never raises — a DB hiccup must not fail the operator's command; it only
    means the setting won't be recalled on the next restart.
    """
    try:
        with _config_session() as s:
            save_config(s, key, payload)
    except Exception as exc:  # noqa: BLE001 - persistence is non-critical
        logger.warning("Persisting %r failed (non-fatal): %s", key, exc)


power_supply_service = PowerSupplyService(
    plc_client=plc_client, logger=logger, session_factory=_config_session
)
temperature_service = TemperatureService(
    plc_client=plc_client, logger=logger, session_factory=_config_session
)

# The SCADA-authoritative persisted routing (interlocks/alarm setpoints + PID).
# Held in memory so the PLC-connect read can't clobber the operator's alarm
# setpoints: _publish_active_config overlays these interlocks onto whatever the
# PLC returns. Loaded on startup and refreshed on every deploy.
_persisted_routing: RoutingConfig | None = None

# Historian write throttle: decouples how often scans are *persisted* from how
# often the PLC is *polled*, so the capture DB grows at an operator-chosen rate
# without slowing the control/stream loop. Runtime-adjustable via /api/capture/config.
capture_throttle = CaptureThrottle(settings.capture_interval_s)

# Global performance mode: switches the scan-loop cadence between the fast
# (performance) and slow (lightweight) intervals to conserve CPU / HMI load.
# Defaults to lightweight — fast polling is opt-in (and the HMI auto-engages it
# whenever its tab is hidden), so an unattended backend stays easy on the Pi.
perf_controller = PerformanceController(
    settings.poll_interval_s,
    settings.lightweight_poll_interval_s,
    mode=PerformanceMode.LIGHTWEIGHT,
)


def _throttled_log_scan(
    session: Session,
    tags: dict[str, float],
    *,
    signal_frame: SignalFrame | None = None,
) -> int:
    """Persist a scan to the historian only when the throttle says it's due."""
    return (
        historian.log_scan(session, tags, signal_frame=signal_frame)
        if capture_throttle.due()
        else 0
    )


class ConnectionManager:
    """Manages WebSocket client connections and broadcasts live updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.register_accepted(websocket)
        logger.info("New WebSocket client connected.")

    def register_accepted(self, websocket: WebSocket) -> None:
        """Register an already-accepted socket without re-accepting it.

        Used by the frame-authenticated path, which must ``accept()`` before it
        can read the credential frame and therefore cannot call ``connect()``.
        Routing the registration through the manager keeps connection
        bookkeeping in one place instead of reaching into
        ``active_connections`` directly.
        """
        self.active_connections.append(websocket)

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

# Latest broadcast frame, cached so HTTP-only clients (e.g. embedded webviews
# that can't hold a WebSocket) can poll /api/snapshot as a streaming fallback.
latest_frame: dict[str, Any] = {}

POLL_FAILURE_ESCALATION_THRESHOLD = 3
POLL_FAILURE_MAX_BACKOFF_S = 5.0

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


control_context = SystemState(alarm_engine_factory=build_alarm_engine)
control_context.attach_clients(plc_client, backup_simulator)
professional_alarm_service = AlarmService(
    manager_from_routing(control_context.active_config)
)


def _apply_control_config(config: RoutingConfig) -> None:
    """Synchronize proven controls and the supervisory alarm workspace."""
    alarm_manager = manager_from_routing(config)
    control_context.apply_config(config, plc_client, backup_simulator)
    professional_alarm_service.reconfigure(alarm_manager)


async def _deploy_approved_routing(config: RoutingConfig) -> None:
    """Deploy one approved revision before publishing it to runtime readers."""
    if not isinstance(config, RoutingConfig):
        raise TypeError("config must be a RoutingConfig")
    if plc_client.connected:
        if not await plc_client.write_routing(config):
            raise RuntimeError("PLC rejected the approved configuration")
        if not await plc_client.save_to_flash():
            raise RuntimeError("PLC configuration was not saved to flash")
    if not await backup_simulator.write_routing(config):
        raise RuntimeError("simulator rejected the approved configuration")
    if not await backup_simulator.save_to_flash():
        raise RuntimeError("simulator configuration was not saved")
    _apply_control_config(config)
    global _persisted_routing
    _persisted_routing = config


configuration_workflow = ConfigurationWorkflow(
    SqliteRevisionRepository(_config_session),
    _deploy_approved_routing,
)


async def modbus_connect_background() -> None:
    """Periodically attempts to connect to PLC in background without blocking polling loop."""
    logger.info("Starting background PLC connection task...")
    while not shutdown_event.is_set():
        try:
            await _connect_once(
                plc=plc_client,
                power_supply=power_supply_service,
                temperature=temperature_service,
                apply_config=_publish_active_config,
                estop_active=control_context.e_stop_active,
            )
        except Exception as e:
            logger.debug(f"Background PLC connect attempt failed: {e}")
        await asyncio.sleep(settings.connect_retry_interval_s)


def _publish_active_config(config: RoutingConfig) -> None:
    """Publish a PLC routing config to the shared control context.

    The interlocks (alarm setpoints) are a SCADA-layer concern; if an operator's
    persisted alarm setpoints exist, they are overlaid onto whatever the PLC
    returned so a stale/default PLC read can never silently reset them.
    """
    if _persisted_routing is not None:
        config = config.model_copy(update={"interlocks": _persisted_routing.interlocks})
    _apply_control_config(config)


def require_read_auth(
    api_key: str | None = Security(_read_api_key_header),
) -> None:
    """Optional gate for the historian/plant read surface.

    Enforces :func:`require_api_key` only when ``P1AM_REQUIRE_READ_AUTH`` is
    enabled. When the setting is off (the default) this is a no-op so the read
    endpoints stay public and the bench HMI keeps working unchanged. The
    existing ``P1AM_DEV_NO_AUTH`` bypass still applies via ``require_api_key``.

    The env var is read per-request (not the ``lru_cache``d settings singleton)
    so the gate can be toggled without a process restart.
    """
    if os.environ.get("P1AM_REQUIRE_READ_AUTH", "").strip().lower() not in _TRUTHY:
        return
    require_api_key(api_key)


def _reject_output_write_if_estopped() -> None:
    if control_context.e_stop_active:
        raise HTTPException(
            status_code=409,
            detail="E-stop active; output writes are inhibited.",
        )


def _require_latest_tag(tag_name: str, *, role: str) -> float:
    try:
        return float(control_context.latest_tags[tag_name])
    except KeyError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Configured {role} tag '{tag_name}' is not mapped.",
        ) from exc


async def poll_plc_loop() -> None:
    """Background loop polling the PLC tags at 10Hz.

    Saves data to DB and streams updates to WS.
    """
    global latest_frame
    logger.info("Starting background PLC polling loop...")
    consecutive_failures = 0
    while not shutdown_event.is_set():
        retry_delay = perf_controller.poll_interval_s
        try:
            frame = await _poll_once(
                plc=plc_client,
                backup=backup_simulator,
                latest_tag_values=control_context.latest_tags,
                ws=ws_manager,
                alicats=alicat_manager,
                power_supply=power_supply_service,
                temperature=temperature_service,
                alarm_engine=control_context.alarm_engine,
                active_alarm_map=control_context.active_alarms,
                session_factory=get_session,
                estop_active=control_context.e_stop_active,
                log_scan=_throttled_log_scan,
            )
            # Cache the frame for the /api/snapshot polling fallback. Reassigning
            # the reference is atomic, so a concurrent reader sees a whole frame.
            if frame:
                latest_frame = frame
                quality = frame.get("comms_health", {}).get("quality")
                if quality in {"good", "uncertain", "simulated"}:
                    professional_alarm_service.observe(frame.get("tags_dict", {}))
            consecutive_failures = 0
        except Exception as loop_err:
            consecutive_failures += 1
            retry_delay = min(
                settings.poll_interval_s * (2 ** (consecutive_failures - 1)),
                POLL_FAILURE_MAX_BACKOFF_S,
            )
            if consecutive_failures < POLL_FAILURE_ESCALATION_THRESHOLD:
                logger.error(f"Unexpected error in PLC polling loop: {loop_err}")
            elif consecutive_failures == POLL_FAILURE_ESCALATION_THRESHOLD:
                logger.warning(
                    "PLC polling loop degraded after %d consecutive failures; "
                    "retrying in %.3fs: %s",
                    consecutive_failures,
                    retry_delay,
                    loop_err,
                )
            else:
                logger.debug(
                    "PLC polling loop still degraded after %d consecutive failures; "
                    "retrying in %.3fs: %s",
                    consecutive_failures,
                    retry_delay,
                    loop_err,
                )
            if consecutive_failures >= POLL_FAILURE_ESCALATION_THRESHOLD:
                latest_frame = {
                    "polling_status": {
                        "status": "degraded",
                        "consecutive_failures": consecutive_failures,
                        "retry_delay_s": retry_delay,
                        "last_error": str(loop_err),
                    }
                }
        # Sleep to maintain 10Hz frequency (100ms cycle)
        await asyncio.sleep(retry_delay)
    logger.info("Background PLC polling loop stopped.")


def _restore_persisted_settings(session: Session) -> None:
    """Recall operator settings from the config store on startup (settings only).

    Restores the SCADA routing (interlocks/alarm setpoints + PID), the heater and
    power-supply configs + last setpoints, the historian capture rate and the
    performance mode. SAFETY: nothing here arms or energizes an output — the
    controllers stay IDLE; the operator presses Start to resume to the recalled
    setpoint. Best-effort per item so a bad/legacy blob only falls back to a
    default, never blocks the boot of the safety-critical controller.
    """
    global _persisted_routing
    try:
        active_revision = configuration_workflow.active()
        routing = (
            active_revision.payload
            if active_revision is not None
            else load_model(session, "routing", RoutingConfig)
        )
        if routing is not None:
            _persisted_routing = routing
            _apply_control_config(routing)
            logger.info("Recalled de-energized configuration settings.")
    except Exception as exc:  # noqa: BLE001 - never block boot on a bad blob
        logger.warning("Routing recall skipped: %s", exc)
    try:
        temperature_service.restore_persisted(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Temperature settings recall skipped: %s", exc)
    try:
        power_supply_service.restore_persisted(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Power-supply settings recall skipped: %s", exc)
    try:
        cap = load_config(session, "capture")
        if cap and "interval_s" in cap:
            capture_throttle.set_interval_s(float(cap["interval_s"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Capture-interval recall skipped: %s", exc)
    try:
        perf = load_config(session, "performance")
        if perf and "mode" in perf:
            perf_controller.set_mode(PerformanceMode(str(perf["mode"])))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Performance-mode recall skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: initialize database and start PLC polling thread & Alicat manager
    init_db()
    with Session(engine) as session:
        load_tags_into_plc_clients(session)
        _restore_persisted_settings(session)
    shutdown_event.clear()
    alicat_manager.start()
    connect_task = asyncio.create_task(modbus_connect_background())
    polling_task = asyncio.create_task(poll_plc_loop())
    retention_task = asyncio.create_task(
        historian_retention_loop(
            shutdown_event=shutdown_event,
            engine=engine,
            logger=logger,
            settings=settings,
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
app.state.control_context = control_context
app.include_router(create_identity_router(identity_service))
app.include_router(create_audit_router(get_session, require_engineer_key))
app.include_router(
    create_alarm_router(
        professional_alarm_service,
        operator_dependency=require_api_key,
        engineer_dependency=require_engineer_key,
    )
)
app.include_router(
    create_configuration_router(
        configuration_workflow,
        engineer_dependency=require_engineer_key,
        admin_dependency=require_admin_key,
    )
)
app.include_router(create_power_supply_router(power_supply_service))
app.include_router(create_temperature_router(temperature_service))

# Data Explorer analysis suite (historian querying, filtering, correlation,
# spectral, trendlines, PCA, export). It is numpy-backed; if numpy or the module
# is unavailable (e.g. the slim image without it) the feature simply stays off
# and the safety-critical control core still boots — mirroring the tools_core
# graceful fallback above.
try:
    from data_explorer_router import create_data_explorer_router

    app.include_router(
        create_data_explorer_router(get_session, read_auth_dep=require_read_auth)
    )
except Exception as exc:  # pragma: no cover - only when numpy/module absent
    logger.warning("Data Explorer disabled (%s): %s", type(exc).__name__, exc)

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


def _audit_principal(request: Request) -> Principal | None:
    return resolve_optional_principal(
        request.headers.get(CREDENTIAL_HEADER_NAME),
        request.headers.get("Authorization"),
    )


def _configuration_revision() -> str:
    active = configuration_workflow.active()
    if active is not None and active.activation_identity:
        return active.activation_identity
    return os.environ.get("P1AM_CONFIG_REVISION", "unversioned")


app.add_middleware(
    MutationAuditMiddleware,
    engine=engine,
    principal_resolver=_audit_principal,
    configuration_revision=_configuration_revision,
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
    config: RoutingConfig | None = await plc_client.read_routing()
    if config is None:
        config = await backup_simulator.read_routing()
    if config is None:
        raise HTTPException(
            status_code=500, detail="Failed to read routing configuration."
        )
    return config


@app.post("/api/routing", dependencies=[Depends(require_admin_key)])
async def update_routing(config: RoutingConfig) -> dict[str, str]:
    """Reject the retired direct-activation path without applying the payload."""
    del config
    raise HTTPException(
        status_code=409,
        detail=(
            "Direct configuration activation is disabled; use the protected "
            "draft, validation, review, approval, and activation workflow."
        ),
    )


# NOTE: E-stop *activation* is intentionally left unauthenticated so a panic
# stop is always reachable even without a credential (safety over confidentiality).
# Clearing the E-stop (below) requires the admin credential.
@app.post("/api/estop")
async def trigger_estop() -> dict[str, str]:
    """Immediate safety shutdown command, zeroing all tag variables."""
    control_context.engage_estop()
    # Latch the controllers FIRST so the next poll cycle cannot re-command a
    # setpoint / re-close the heater relay and re-energize after we zero below.
    power_supply_service.engage_estop()
    temperature_service.engage_estop()
    if not plc_client.connected:
        await backup_simulator.trigger_estop()
        control_context.reset_tag_values()
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
    turns green while the plant is still tripped. The server-side control
    context latch is only lowered once the controller (or the backup simulator,
    when the PLC is offline) acknowledges the reset.
    """
    if not plc_client.connected:
        cleared = await backup_simulator.clear_estop()
        if not cleared:
            raise HTTPException(
                status_code=502,
                detail="Backup simulator did not acknowledge the E-stop reset.",
            )
        control_context.clear_estop()
        # Release the controller latches in sim mode too, otherwise they stay
        # engaged and refuse to re-arm after the operator clears the E-stop.
        power_supply_service.clear_estop()
        temperature_service.clear_estop()
        return {
            "status": "success",
            "message": "Simulated E-stop cleared.",
        }

    cleared = await plc_client.clear_estop()
    if not cleared:
        # Leave the E-stop latch set so the HMI keeps showing the tripped state.
        raise HTTPException(
            status_code=502,
            detail=("PLC did not acknowledge the E-stop reset; plant remains tripped."),
        )
    await backup_simulator.clear_estop()
    control_context.clear_estop()
    # Release the controller latches too. They return to IDLE with permissive
    # off, so the operator must deliberately re-arm before any output can flow.
    power_supply_service.clear_estop()
    temperature_service.clear_estop()
    return {"status": "success", "message": "Hardware E-stop cleared."}


@app.get("/api/snapshot", dependencies=[Depends(require_read_auth)])
async def get_snapshot() -> dict[str, Any]:
    """Latest live frame — identical shape to the /api/stream WebSocket frames.

    Returns the cached frame from the poll loop (no PLC round-trip), so HTTP-only
    clients that can't hold a WebSocket — e.g. an embedded VS Code Simple Browser
    webview — can poll this as a streaming fallback and still see live data.
    """
    return latest_frame


@app.get("/api/alarms/active")
async def get_active_alarms() -> list[dict[str, Any]]:
    """Get all currently active or unacknowledged alarms."""
    return list(control_context.active_alarms.values())


@app.post(
    "/api/alarms/{tag_id}/acknowledge",
    dependencies=[Depends(require_api_key)],
)
async def acknowledge_alarm(
    tag_id: str,
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, str]:
    """Acknowledge a specific active alarm."""
    if tag_id in control_context.active_alarms:
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
            raise HTTPException(
                status_code=500,
                detail=f"Failed to persist acknowledgment for alarm {tag_id}.",
            ) from e

        if not control_context.acknowledge_alarm(tag_id):
            raise HTTPException(
                status_code=409,
                detail=f"Alarm {tag_id} could not be acknowledged.",
            )

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


@app.get("/api/events", dependencies=[Depends(require_read_auth)])
def get_events(
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


def _trend_signal_metadata(
    db: Session,
    tag_name: str,
    sample_times: list[datetime],
) -> dict[str, list[Any]]:
    if not sample_times:
        return {
            "qualities": [],
            "diagnostic_reasons": [],
            "source_timestamps": [],
            "sequences": [],
            "sources": [],
        }
    rows = db.exec(
        select(TagLog)
        .where(col(TagLog.tag_name) == tag_name)
        .where(col(TagLog.timestamp).in_(sample_times))
    ).all()
    by_timestamp = {row.timestamp: row for row in rows}
    ordered = [by_timestamp[timestamp] for timestamp in sample_times]
    return {
        "qualities": [row.quality for row in ordered],
        "diagnostic_reasons": [row.diagnostic_reason for row in ordered],
        "source_timestamps": [
            (row.source_timestamp or row.timestamp).isoformat() for row in ordered
        ],
        "sequences": [row.sequence for row in ordered],
        "sources": [row.source for row in ordered],
    }


@app.get("/api/trends", dependencies=[Depends(require_read_auth)])
def get_trends(
    tag_id: str,
    start_time: str,
    end_time: str,
    smoothing: str = "none",
    window_size: int = 5,
    alpha: float = 0.2,
    max_points: int = TRENDS_MAX_POINTS,
    db: Session = Depends(get_session),  # noqa: B008
) -> dict[str, Any]:
    """Fetch historical trends, decimated to span the whole requested window.

    The series is downsampled server-side to at most ``max_points`` samples that
    evenly span ``[start_time, end_time]``, so a multi-hour or multi-day request
    returns the entire span instead of only its newest slice. ``truncated`` is
    True whenever decimation occurred. Any server-side ``smoothing`` is applied
    to the decimated values. ``max_points`` is clamped to a sane range; an
    out-of-range value yields HTTP 400.
    """
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

    try:
        sample_times, values, truncated = query_trend_series(
            db,
            tag_name=tag_name,
            start=start_dt,
            end=end_dt,
            max_points=max_points,
        )
    except (TypeError, ValueError) as query_err:
        raise HTTPException(status_code=400, detail=str(query_err)) from query_err

    timestamps = [ts.isoformat() for ts in sample_times]

    if smoothing == "moving_average" and values:
        values = moving_average(values, window_size)
    elif smoothing == "exponential_smoothing" and values:
        values = exponential_smoothing(values, alpha)

    return {
        "timestamps": timestamps,
        "values": values,
        **_trend_signal_metadata(db, tag_name, sample_times),
        "truncated": truncated,
    }


@app.get("/api/export", dependencies=[Depends(require_read_auth)])
def export_data(
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
def get_capture_status(
    db: Session = Depends(get_session),  # noqa: B008
) -> CaptureStats:
    """Report the captured historian: rows, time span, distinct tags, disk size.

    Capture is automatic — the polling loop logs every scan whenever the backend
    is up — so ``capturing`` mirrors that always-on behavior for the HMI's REC
    indicator.
    """
    return capture_stats(db, capturing=True)


@app.get("/api/capture/config", response_model=CaptureConfig)
async def get_capture_config() -> CaptureConfig:
    """Return the current historian sampling interval (seconds between writes)."""
    return CaptureConfig(interval_s=capture_throttle.interval_s)


@app.put(
    "/api/capture/config",
    response_model=CaptureConfig,
    dependencies=[Depends(require_admin_key)],
)
async def update_capture_config(req: CaptureConfig) -> CaptureConfig:
    """Set how often scans are persisted. Larger interval => smaller data files.

    Takes effect immediately for the running scan loop; admin-gated since it
    changes the historian's data-retention characteristics.
    """
    try:
        capture_throttle.set_interval_s(req.interval_s)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Capture interval set to %.3f s", capture_throttle.interval_s)
    _persist_setting("capture", {"interval_s": capture_throttle.interval_s})
    return CaptureConfig(interval_s=capture_throttle.interval_s)


class PerformanceModeRequest(BaseModel):
    """Operator selection of the global performance mode."""

    mode: PerformanceMode


@app.get("/api/performance", response_model=PerformanceConfig)
async def get_performance() -> PerformanceConfig:
    """Return the active performance mode + its resolved poll interval."""
    return perf_controller.config()


@app.put(
    "/api/performance",
    response_model=PerformanceConfig,
    dependencies=[Depends(require_admin_key)],
)
async def update_performance(req: PerformanceModeRequest) -> PerformanceConfig:
    """Switch performance/lightweight mode. Takes effect on the next scan."""
    try:
        perf_controller.set_mode(req.mode)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "Performance mode set to %s (poll %.3f s)",
        perf_controller.mode,
        perf_controller.poll_interval_s,
    )
    _persist_setting("performance", {"mode": str(perf_controller.mode)})
    return perf_controller.config()


class CaptureClearRequest(BaseModel):
    """Operator request to clear captured data."""

    include_events: bool = False


@app.post(
    "/api/capture/clear",
    response_model=ClearResult,
    dependencies=[Depends(require_admin_key)],
)
def clear_capture_data(
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


def _latest_tag_or_http_error(tag_name: str, role: str, pid_index: int) -> float:
    """Return the latest tag value or raise a descriptive PID tuning error."""
    try:
        return float(control_context.latest_tags[tag_name])
    except KeyError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"PID loop {pid_index} {role} tag '{tag_name}' is not mapped in "
                "the latest tag values. Check PLC routing before tuning."
            ),
        ) from exc


@app.post("/api/tags/{tag_id}", dependencies=[Depends(require_admin_key)])
async def write_tag_value(tag_id: str, payload: TagWritePayload) -> dict[str, str]:
    """Manually force/write a 32-bit float value directly to a tag register."""
    _reject_output_write_if_estopped()
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
        control_context.write_tag(tag_name, payload.value)
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

    control_context.write_tag(tag_name, payload.value)
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

    # Reject a double-start rather than silently overwriting an in-progress
    # session (a double-click or race would otherwise wipe the captured initial
    # PV/CV and step history). The operator must stop the loop first.
    if pid_index in control_context.tuning_sessions:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Tuning session already active for PID loop {pid_index}; "
                "stop it before starting a new one."
            ),
        )

    pv_tag = control_context.active_config.pids[pid_index].pv_tag
    cv_tag = control_context.active_config.pids[pid_index].cv_tag
    current_pv = _latest_tag_or_http_error(pv_tag, "PV", pid_index)
    current_cv = _latest_tag_or_http_error(cv_tag, "CV", pid_index)

    control_context.tuning_sessions[pid_index] = {
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
    if pid_index not in control_context.tuning_sessions:
        raise HTTPException(
            status_code=400, detail="Tuning session not active for this PID loop."
        )

    _reject_output_write_if_estopped()
    session = control_context.tuning_sessions[pid_index]
    cv_tag = control_context.active_config.pids[pid_index].cv_tag
    initial_cv = _latest_tag_or_http_error(cv_tag, "CV", pid_index)

    session["step_triggered"] = True
    session["step_time"] = time.time() - session["start_time"]
    session["initial_cv"] = initial_cv
    session["final_cv"] = payload.step_value

    await plc_client.write_tag(cv_tag, payload.step_value)
    await backup_simulator.write_tag(cv_tag, payload.step_value)
    control_context.write_tag(cv_tag, payload.step_value)

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
    if pid_index not in control_context.tuning_sessions:
        raise HTTPException(
            status_code=400, detail="Tuning session not active for this PID loop."
        )

    session = control_context.tuning_sessions.pop(pid_index)
    result = identify_fopdt_and_tune(
        session["history"],
        step_triggered=session["step_triggered"],
        initial_pv=session["initial_pv"],
        initial_cv=session["initial_cv"],
        final_cv=session["final_cv"],
        step_time=session["step_time"],
    )
    return cast(dict[str, Any], result.as_response())


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
    return cast(dict[str, Any], simulate_pid_vs_mpc(payload))


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
        ws_manager.register_accepted(websocket)
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


@app.get(
    "/api/project/ladder-explorer",
    dependencies=[Depends(require_read_auth)],
)
def get_ladder_explorer(
    db: Session = Depends(get_session),  # noqa: B008
) -> list[dict[str, Any]]:
    """Retrieve all tag definitions with their PLC register mappings for exploration."""
    areas = db.exec(select(PlantArea)).all()
    units = db.exec(select(PlantUnit)).all()
    equips = db.exec(select(PlantEquipment)).all()
    tags = db.exec(select(TagDefinitionDb)).all()
    area_by_id = {area.id: area for area in areas}
    unit_by_id = {unit.id: unit for unit in units}
    equip_by_id = {equip.id: equip for equip in equips}

    results = []
    for t in tags:
        equip = equip_by_id.get(t.equipment_id) if t.equipment_id else None
        unit = unit_by_id.get(equip.unit_id) if equip else None
        area = area_by_id.get(unit.area_id) if unit else None

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


@app.get("/api/plant", dependencies=[Depends(require_read_auth)])
def get_plant_hierarchy_api(
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
