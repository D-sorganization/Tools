#!/usr/bin/env bash
#
# Install the "P1AM Heater Control" desktop launcher for the current user:
#   - a double-clickable icon on the Desktop, and
#   - a searchable entry in the applications menu.
#
# No root required. Re-run any time to refresh (e.g. after moving the repo).
# Paths are detected from this script's location, so it isn't pinned to one
# machine or user.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch-hmi.sh"
ICON="$SCRIPT_DIR/p1am-hmi.svg"

[ -f "$LAUNCHER" ] || { echo "ERROR: launcher not found: $LAUNCHER" >&2; exit 1; }
[ -f "$ICON" ]     || { echo "ERROR: icon not found: $ICON" >&2; exit 1; }
chmod +x "$LAUNCHER"

# Resolve the (possibly localized) Desktop directory.
if [ -f "$HOME/.config/user-dirs.dirs" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/user-dirs.dirs" 2>/dev/null || true
fi
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR" "$APPS_DIR"

entry="[Desktop Entry]
Version=1.0
Type=Application
Name=P1AM Heater Control
GenericName=Heater Control HMI
Comment=Start the P1AM control system and open the HMI (localhost:3002)
Exec=$LAUNCHER
Icon=$ICON
Terminal=false
Categories=Utility;Engineering;
StartupNotify=true
Keywords=P1AM;heater;PLC;HMI;SCADA;control;temperature;"

app_file="$APPS_DIR/p1am-hmi.desktop"
desk_file="$DESKTOP_DIR/p1am-hmi.desktop"
printf '%s\n' "$entry" > "$app_file"
printf '%s\n' "$entry" > "$desk_file"
chmod +x "$app_file" "$desk_file"

# Mark the desktop copy trusted so PIXEL/GNOME launch it without the
# "untrusted launcher" prompt (best-effort — harmless if gio is absent).
gio set "$desk_file" metadata::trusted true 2>/dev/null || true
update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo "Installed the 'P1AM Heater Control' launcher:"
echo "  Desktop icon : $desk_file"
echo "  App menu     : $app_file"
echo "  Runs         : $LAUNCHER"
echo
echo "Double-click the desktop icon to start the system and open the HMI."
