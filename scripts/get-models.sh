#!/bin/bash
# Downloads the models listed in models/models.txt into <ROOT>/models (default ~/.local/share/frametint/models).
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
ROOT="${FT_HOME:-$HOME/.local/share/frametint}"; DEST="$ROOT/models"; mkdir -p "$DEST"
n=0; bad=0
while read -r name url sha; do
  case "$name" in ""|\#*) continue ;; esac
  n=$((n+1))
  if [ -f "$DEST/$name" ] && echo "$sha  $DEST/$name" | sha256sum -c --quiet - 2>/dev/null; then echo "have $name"; continue; fi
  echo "downloading $name ..."
  curl -fL --retry 3 -o "$DEST/$name.part" "$url" || { echo "PROBLEM: could not download $name"; bad=1; rm -f "$DEST/$name.part"; continue; }
  if echo "$sha  $DEST/$name.part" | sha256sum -c --quiet -; then mv "$DEST/$name.part" "$DEST/$name"; else echo "PROBLEM: checksum mismatch for $name"; rm -f "$DEST/$name.part"; bad=1; fi
done < models/models.txt
[ "$n" -eq 0 ] && echo "No models are listed in models/models.txt yet. See models/README.md to get or make them."
exit $bad
