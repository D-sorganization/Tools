# P1AM control system — systemd deployment

Runs the control system as two systemd services so it **starts on boot** and
**auto-restarts on failure** — surviving reboots, power loss, and terminal/SSH
session teardown (no more manual `run_pi.sh` relaunches).

| Service              | What it runs                                   | Port            |
| -------------------- | ---------------------------------------------- | --------------- |
| `p1am-backend`       | FastAPI/uvicorn — the single Modbus master     | `127.0.0.1:8000` |
| `p1am-frontend`      | HMI production build served by `vite preview`  | `:3002`         |

The backend binds localhost only; the HMI proxies `/api` + `/ws` to it. The
backend is intentionally a **single process** (one worker) — the firmware is a
single-client Modbus master, so multiple workers would corrupt the link.

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
  code, `sudo systemctl restart p1am-backend`. The `p1am-frontend` unit runs
  `npm run build` then `vite preview` (serving the production build, **not** a
  hot-reloading dev server), so after editing HMI code
  `sudo systemctl restart p1am-frontend` — it rebuilds on start (~30 s).
- **Remote access:** Raspberry Pi Connect, VNC-over-Tailscale
  (`100.108.70.33:5900`), and SSH — see the top-level `README.md`.
- **Network/PLC at boot:** the backend retries the PLC connection in the
  background, so it comes up cleanly even if the network or PLC is slow to
  appear after a power cycle.
- The host's static IP (`192.168.1.50/24` on `eth0`) is configured separately
  (persists across reboots) and is not managed by these units.
