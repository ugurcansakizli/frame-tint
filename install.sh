#!/bin/bash
# Frame Tint installer. Run ON THE STEAM FRAME (SteamOS, e.g. in Desktop Mode's terminal):   bash install.sh
# Needs no root. Everything goes into ~/.local/share/frametint (override: FT_HOME=/some/folder bash install.sh).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="${FT_HOME:-$HOME/.local/share/frametint}"
BIN="$HOME/.local/bin"; APPS="$HOME/.local/share/applications"; CFG="$HOME/.config/frametint"
VRPATHREG=/opt/steamvr/bin/linuxarm64/vrpathreg
die() { echo "PROBLEM: $*" >&2; exit 1; }

[ "$(uname -m)" = aarch64 ] && [ -x "$VRPATHREG" ] || die "this is for the Steam Frame: it needs SteamOS on aarch64 with SteamVR in /opt/steamvr"
for t in g++ gcc python3 curl; do command -v "$t" >/dev/null || die "'$t' is missing"; done
[ -e /persist/xrservice.json ] || echo "note: /persist/xrservice.json (the headset's lens calibration) was not found; Frame Tint will not run without it"

echo "== building (about a minute)"
bash scripts/build.sh

echo "== installing into $ROOT"
mkdir -p "$ROOT"/{bin,lib,models,share,driver/frametint/bin/linuxarm64} "$BIN" "$APPS" "$CFG"
install -m 755 build/ftgrab build/ftoverlay "$ROOT/bin/"
install -m 755 build/driver_frametint.so "$ROOT/driver/frametint/bin/linuxarm64/"
cat > "$ROOT/driver/frametint/driver.vrdrivermanifest" <<'EOF'
{
	"alwaysActivate": true,
	"name": "frametint",
	"directory": "",
	"resourceOnly": false,
	"hmd_presence": []
}
EOF
cp -f src/server/*.py "$ROOT/lib/"
cp -rf share/. "$ROOT/share/"
[ -f "$CFG/frametint.conf" ] || cp config/frametint.conf "$CFG/frametint.conf"      # an existing settings file is kept
install -m 755 launcher/frametint-start launcher/frametint-stop "$BIN/"
sed "s#@BIN@#$BIN#; s#@ROOT@#$ROOT#" launcher/frametint.desktop.in > "$APPS/frametint.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true

echo "== python environment (needs internet)"
[ -x "$ROOT/venv/bin/python" ] || python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install --quiet --upgrade pip
"$ROOT/venv/bin/pip" install --quiet -r requirements.txt

echo "== registering the camera driver with SteamVR"
STUB="$(mktemp -d)"; printf '#!/bin/sh\nexit 0\n' > "$STUB/xdg-open"; chmod 755 "$STUB/xdg-open"   # vrpathreg tries to open a URL; silence it
PATH="$STUB:$PATH" "$VRPATHREG" adddriver "$ROOT/driver/frametint" >/dev/null 2>&1 || { rm -rf "$STUB"; die "vrpathreg adddriver failed"; }
rm -rf "$STUB"

echo "== models"
for m in models/*.onnx; do [ -f "$m" ] && cp -f "$m" "$ROOT/models/"; done
FT_HOME="$ROOT" bash scripts/get-models.sh || echo "note: some models could not be downloaded; see models/README.md"

cat <<EOF

DONE. Two things left:
  1. Restart SteamVR once so it loads the camera driver (this reloads the whole headset session, so save anything open):
       XDG_RUNTIME_DIR=/run/user/\$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/\$(id -u)/bus systemctl --user restart steamvr.service
  2. Put the headset on and start "Frame Tint" from SteamVR's Launch Program list or the desktop app menu.
Settings: $CFG/frametint.conf     Log: ~/.local/state/frametint/frametint.log     Models: $ROOT/models
EOF
