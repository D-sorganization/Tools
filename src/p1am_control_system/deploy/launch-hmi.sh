#!/usr/bin/env bash
#
# P1AM Heater Control — operator launcher.
#
# Double-click the "P1AM Heater Control" desktop icon (installed by
# install-desktop-launcher.sh) to run this. No terminal or editor needed. It:
#   1. makes sure the control services are running (they normally auto-start on
#      boot; this recovers a stopped one),
#   2. waits for the HMI to come up on http://localhost:3002, then
#   3. opens it in a clean, full-window browser.
#
# The backend connects to the PLC on its own and the HMI shows the live link
# state, so this launcher never blocks on the PLC. Safe to run repeatedly.
#
# Handy overrides:
#   P1AM_KIOSK=1            full-screen kiosk (no window chrome) instead of an app window
#   P1AM_LAUNCH_TIMEOUT=90  seconds to wait for the HMI before giving up
#   P1AM_LAUNCH_NO_BROWSER=1  only check readiness and print the URL (for testing)
#
set -uo pipefail

HMI_URL="http://localhost:3002"
BACKEND_URL="http://localhost:8000/api/performance"
SERVICES=(p1am-backend p1am-frontend)
WAIT_SECONDS="${P1AM_LAUNCH_TIMEOUT:-90}"

have() { command -v "$1" >/dev/null 2>&1; }

# Show a message the operator can see: a GUI dialog on a desktop, else stdout.
dialog() { # dialog <info|error|warning> <text>
  local kind="$1" text="$2"
  if have zenity; then
    zenity --"$kind" --no-wrap --title "P1AM Heater Control" --text "$text" 2>/dev/null || true
  fi
  if [ "$kind" = "error" ]; then echo "ERROR: $text" >&2; else echo "$text"; fi
}

# 1) Ensure the services are up. They auto-start on boot; this only matters if
#    one was stopped. Starting a system service needs sudo — the Pi's default
#    user has passwordless sudo, so this is silent; if it fails we simply wait,
#    since systemd will have started them anyway.
for svc in "${SERVICES[@]}"; do
  if ! systemctl is-active --quiet "$svc" 2>/dev/null; then
    sudo -n systemctl start "$svc" >/dev/null 2>&1 || true
  fi
done

# 2) Wait for the HMI to answer (the frontend rebuilds on a fresh boot, so give
#    it time). curl -f treats a 200 as success.
ready=0
for _ in $(seq 1 "$WAIT_SECONDS"); do
  if curl -fsS -o /dev/null --max-time 2 "$HMI_URL" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  dialog error "The HMI did not start within ${WAIT_SECONDS}s at ${HMI_URL}.

Try again in a moment, or check the services from a terminal:
  systemctl status ${SERVICES[*]}
  journalctl -u p1am-frontend -e"
  exit 1
fi

# 2b) Seed the HMI's credential.
#
# The backend now requires an API key on every control action and on the
# telemetry WebSocket (issue #4007). The HMI stores the key per browser profile;
# hand it over once via the URL *fragment*, which is client-side only — it is
# never sent to any server and the HMI strips it from the address bar on load.
# Falls back to the plain URL when the env file is unreadable (e.g. a --bench
# install, which has no credential at all), and the HMI then prompts.
ENV_FILE="${P1AM_ENV_FILE:-/etc/p1am/backend.env}"
if [ -r "$ENV_FILE" ]; then
  api_key="$(sed -n 's/^P1AM_API_KEY=//p' "$ENV_FILE" | head -n1)"
  if [ -n "${api_key:-}" ]; then
    HMI_URL="${HMI_URL}/#apikey=${api_key}"
  fi
fi

# The HMI is up. Note (non-fatal) if the backend/PLC link isn't answering yet —
# the HMI itself shows the connection state, so we still open it.
if ! curl -fsS -o /dev/null --max-time 2 "$BACKEND_URL" 2>/dev/null; then
  echo "Note: HMI is up but the backend link isn't answering yet — the HMI will show its status."
fi

# Test hook: skip the actual browser launch (used for headless verification).
if [ "${P1AM_LAUNCH_NO_BROWSER:-0}" = "1" ]; then
  echo "READY: $HMI_URL"
  exit 0
fi

# When launched from the desktop the display env is set; default it for safety
# if this is ever run from a bare shell.
export DISPLAY="${DISPLAY:-:0}"

# 3) Open the HMI. Default: a clean maximized app window (no tabs/address bar).
#    Set P1AM_KIOSK=1 for locked full-screen.
kiosk="${P1AM_KIOSK:-0}"
chromium_app=(--app="$HMI_URL" --start-maximized --no-first-run --disable-translate)
chromium_kiosk=(--kiosk "$HMI_URL" --no-first-run --disable-translate)

# Pick a chromium to run. IMPORTANT: prefer the RAW binary over the
# /usr/bin/chromium(-browser) wrapper. On Raspberry Pi OS that wrapper can inject
# a malformed default flag (e.g. a bare `--no-decommit-pooled-pages` instead of
# `--js-flags=...`) that the browser rejects, so the launch aborts and the
# desktop icon appears to "do nothing". The raw binary ignores the flags config
# and starts cleanly.
chromium_bin=""
for cand in /usr/lib/chromium/chromium /usr/lib/chromium-browser/chromium-browser; do
  [ -x "$cand" ] && { chromium_bin="$cand"; break; }
done
if [ -z "$chromium_bin" ]; then
  if have chromium-browser; then chromium_bin="$(command -v chromium-browser)"
  elif have chromium; then chromium_bin="$(command -v chromium)"; fi
fi

if [ -n "$chromium_bin" ]; then
  [ "$kiosk" = "1" ] && exec "$chromium_bin" "${chromium_kiosk[@]}" || exec "$chromium_bin" "${chromium_app[@]}"
elif have firefox; then
  [ "$kiosk" = "1" ] && exec firefox --kiosk "$HMI_URL" || exec firefox "$HMI_URL"
elif have xdg-open; then
  exec xdg-open "$HMI_URL"
else
  dialog error "No web browser (chromium/firefox) found to open ${HMI_URL}."
  exit 1
fi
