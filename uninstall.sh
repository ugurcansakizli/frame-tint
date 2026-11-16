#!/bin/bash
# Removes Frame Tint. Run on the Steam Frame:   bash uninstall.sh
set -u
ROOT="${FT_HOME:-$HOME/.local/share/frametint}"
VRPATHREG=/opt/steamvr/bin/linuxarm64/vrpathreg
"$HOME/.local/bin/frametint-stop" 2>/dev/null
STUB="$(mktemp -d)"; printf '#!/bin/sh\nexit 0\n' > "$STUB/xdg-open"; chmod 755 "$STUB/xdg-open"
[ -x "$VRPATHREG" ] && PATH="$STUB:$PATH" "$VRPATHREG" removedriver "$ROOT/driver/frametint" >/dev/null 2>&1
rm -rf "$STUB"
rm -f "$HOME/.local/bin/frametint-start" "$HOME/.local/bin/frametint-stop" "$HOME/.local/share/applications/frametint.desktop"
rm -rf "$ROOT" "$HOME/.local/state/frametint" "/run/user/$(id -u)/frametint"
echo "Removed. Your settings file ~/.config/frametint/frametint.conf was kept; delete it by hand if you like."
echo "The camera driver stays loaded until SteamVR restarts."
