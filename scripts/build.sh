#!/bin/bash
# Builds the three native programs into build/ (run on the Steam Frame; needs only g++ and gcc).
#   bash scripts/build.sh          build everything
#   bash scripts/build.sh clean
# SteamVR's libopenvr_api.so is used from STEAMVR; OpenVR and Vulkan headers are vendored in third_party/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
CXX="${CXX:-g++}"; CC="${CC:-gcc}"
STEAMVR="${STEAMVR:-/opt/steamvr/bin/linuxarm64}"
[ "${1:-}" = clean ] && { rm -rf build; echo cleaned; exit 0; }
mkdir -p build

echo "driver_frametint.so (loaded into vrserver: lends the camera buffers)"
"$CXX" -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -Isrc/common -Ithird_party/openvr \
  src/driver/driver_frametint.cpp -o build/driver_frametint.so -lpthread

echo "ftgrab (camera grabber)"
"$CC" -O2 -Wall -Wno-unused-function -Isrc/common src/ftgrab/ftgrab.c -o build/ftgrab

echo "ftoverlay (in-headset panel, dashboard panel, controller toggle)"
"$CXX" -O2 -std=c++17 -pthread -Isrc/overlay -Ithird_party/openvr -Ithird_party/vulkan -Ithird_party \
  src/overlay/ftoverlay.cpp src/overlay/canvas.cpp -o build/ftoverlay \
  -L"$STEAMVR" -lopenvr_api -l:libvulkan.so.1 -Wl,-rpath,"$STEAMVR"
echo "built into build/"
