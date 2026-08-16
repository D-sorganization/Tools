# P1AM control system — systemd deployment

Runs the control system as two systemd services so it **starts on boot** and
**auto-restarts on failure** — surviving reboots, power loss, and terminal/SSH
session teardown (no more manual `run_pi.sh` relaunches).

| Service         | What it runs                                  | Bind             |
| --------------- | --------------------------------------------- | ---------------- |
| `p1am-backend`  | FastAPI/uvicorn — the single Modbus master    | `127.0.0.1:8000` |
| `p1am-frontend` | Pre-built HMI bundle served by `vite preview` | `127.0.0.1:3002` |

**Both** bind loopback. The backend was always meant to; the HMI now does too.
`vite preview` proxies `/api` and the telemetry WebSocket straight through to
the backend, so a preview server listening on every interface handed the whole
plant VLAN an unauthenticated path to the "loopback-only" control API —
`curl -X POST http://<pi-ip>:3002/api/estop/clear` reached it from anywhere
(issue #4007). The kiosk browser runs on the Pi, so loopback is all it needs.
To reach the HMI from another machine, forward a port over SSH:

```bash
ssh -L 3002:127.0.0.1:3002 pi@<pi-ip>       # then browse http://localhost:3002
```

The backend is intentionally a **single process** (one worker) — the firmware is
a single-client Modbus master, so multiple workers would corrupt the link.

## Install / update

```bash
./deploy/install-services.sh
```

Idempotent — re-run after pulling changes to refresh the units and restart.
Paths and the service user are detected from the checkout, so it is not pinned
to one machine. Override the PLC address if needed:

```bash
PLC_IP=192.168.1.100 PLC_PORT=502 ./deploy/install-services.sh
```

Requires `sudo` (writes unit files to `/etc/systemd/system`).

The installer now also:

- installs the backend's runtime dependencies from the `p1am` extra in
  `pyproject.toml` and **verifies the app imports** before writing a unit — the
  old script only checked that a Python interpreter existed, so a missing
  `pydantic-settings` or `python-multipart` produced an invisible crash loop
  under `Restart=always` (issue #4014);
- **builds the HMI bundle once, here.** A full TypeScript + Vite build used to
  run inside `ExecStart` on a `Restart=always` unit, saturating all four Pi
  cores for 1–3 minutes on every start and looping forever if preview then
  failed to bind (issue #4036). `ExecStart` is now just `npm run preview`;
- sets scheduling priority on both units. The Modbus master is the real-time
  path (`Nice=-5`, `CPUWeight=800`); the HMI yields to it (`Nice=10`,
  `CPUWeight=100`);
- reports whether the Rust `tools_core` SCADA kernel is present, and builds the
  wheel when a toolchain is available. Without it the backend silently runs the
  pure-Python `scada_fallback`.

## Credentials

The installer generates an operator key and an admin key into a root-owned
environment file and references it from the unit:

```
/etc/p1am/backend.env      root:<service-group>, mode 0640
  P1AM_API_KEY=…           operator tier: telemetry stream, alarm acknowledge
  P1AM_ADMIN_API_KEY=…     admin tier: E-stop clear, tag writes, setpoints,
                           routing deploy, project import, historian clear
```

Existing keys are **preserved** across re-runs, so updating the checkout does
not invalidate a browser or a script that already holds one. The installer
refuses to write a unit if neither key ends up present.

This replaces the previous behaviour, where the unit hardcoded
`Environment=P1AM_DEV_NO_AUTH=1` — which short-circuits `require_api_key`,
`require_admin_key` and the WebSocket gate, so **every production install
shipped with authentication disabled** (issue #4007).

That flag existed for a reason: the HMI had no credential handling at all, so
with authentication on the shipped product did not work. It does now — see
_HMI credential_ below — and the bypass is available only behind an explicit
flag that says what it does:

```bash
./deploy/install-services.sh --bench     # NO AUTH. Isolated bench only.
```

### HMI credential

The HMI stores the key in `localStorage`, scoped to one browser profile. Three
ways it gets there:

1. **Kiosk (automatic).** `deploy/launch-hmi.sh` reads `P1AM_API_KEY` from
   `/etc/p1am/backend.env` and opens the HMI at `#apikey=…`. A URL _fragment_
   is never sent to any server, and the HMI strips it from the address bar on
   load, so nothing lands in a log or in browser history.
2. **Prompt.** If the backend closes the telemetry socket with 1008, the HMI
   asks for the key.
3. **By hand**, e.g. from a laptop over an SSH tunnel:

   ```bash
   sudo grep '^P1AM_API_KEY=' /etc/p1am/backend.env
   ```

   then paste it when prompted.

### Calling the API from a script

Two things changed for non-browser callers (issue #4037):

```bash
curl -X POST http://127.0.0.1:8000/api/estop/clear \
     -H "X-API-Key: $P1AM_ADMIN_API_KEY" \
     -H "X-Requested-With: p1am-hmi"
```

- **Credential.** Reads now require one too; the installer sets
  `P1AM_REQUIRE_READ_AUTH=1`. `GET /api/routing` alone discloses the register
  map, every scale factor and every interlock trip limit.
- **`X-Requested-With`.** State-changing requests must carry a header (or
  `Content-Type: application/json`) that a CORS-"simple" request cannot
  produce. Several control routes take no request body at all, which made them
  reachable from any page the kiosk browser happened to open —
  `fetch(url, {method:"POST", mode:"no-cors"})` hides the response but not the
  effect on a 110 V heater. Requests whose `Origin` is outside
  `P1AM_CORS_ORIGINS` are refused outright.

`POST /api/estop` — the **panic stop** — is exempt from the header requirement
so it stays reachable from a bare shell with no credential and no headers. It
is not exempt from the origin check, so a malicious page still cannot trip the
plant.

## Audit trail

Every state-changing request is recorded in an append-only `auditevent` table
with the route, redacted payload, resolved credential tier, a non-reversible
key fingerprint, the client IP and the response status (issue #4029). It is a
separate table from `EventLog`, so neither the client-writable
`POST /api/events` nor `POST /api/capture/clear {"include_events": true}` can
forge or erase it. Each row is also mirrored to journald:

```bash
journalctl -u p1am-backend -f | grep AUDIT
```

## Manage

```bash
systemctl status p1am-backend p1am-frontend     # state
journalctl -u p1am-backend -f                   # live backend logs
journalctl -u p1am-frontend -f                  # live HMI logs
sudo systemctl restart p1am-backend             # restart after a code change
sudo systemctl stop p1am-frontend               # stop the HMI
sudo systemctl disable --now p1am-backend p1am-frontend   # remove from boot
```

## Notes

- **Live code changes:** neither service auto-reloads. After editing backend
  code, `sudo systemctl restart p1am-backend`. After editing frontend code,
  re-run `./deploy/install-services.sh` (it rebuilds the bundle), or run
  `npm run build` in `frontend/` and restart `p1am-frontend`.
- **Check the boot log after any install.** The backend logs its resolved
  credential posture at startup (issue #4041), so a half-configured deployment
  is visible immediately instead of at the first control action:

  ```bash
  journalctl -u p1am-backend -b | grep 'auth configuration'
  ```

- **Simulated PLC.** An unrecognised or missing `P1AM_PLC_DRIVER` falls back to
  the simulator, which produces live-looking but entirely fabricated process
  values. The backend now prints an unmissable banner when that happens — if
  `SIMULATED PLC — NO REAL HARDWARE IS CONNECTED` appears in the boot log, the
  HMI is not showing the plant.
- **Network/PLC at boot:** the backend retries the PLC connection in the
  background, so it comes up cleanly even if the network or PLC is slow to
  appear after a power cycle.
- The host's static IP (`192.168.1.50/24` on `eth0`) is configured separately
  (persists across reboots) and is not managed by these units.
