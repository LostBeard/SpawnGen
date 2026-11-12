#!/usr/bin/env bash
# Self-contained builds of SpawnGen for people without .NET: the desktop app (the Studio in a native window over the in-process server) and
# the server (open http://localhost:8188 in a browser), per platform, into artifacts/SpawnGen-{desktop,server}-<rid>/.
#   tools/package.sh                      Windows x64
#   tools/package.sh win-x64 linux-x64 osx-arm64
# Each folder runs from anywhere (SpawnGen.exe / SpawnGen, SpawnGen.Server.exe / SpawnGen.Server). Linux's desktop app needs WebKitGTK
# (libwebkit2gtk-4.1); the macOS build is unsigned (run it from a terminal, or allow it in Privacy & Security).
set -euo pipefail
cd "$(dirname "$0")/.."
rids=("$@")
[ ${#rids[@]} -eq 0 ] && rids=(win-x64)
for rid in "${rids[@]}"; do
  dotnet publish src/SpawnGen.Desktop -c Release -r "$rid" --self-contained -o "artifacts/SpawnGen-desktop-$rid" --nologo
  dotnet publish src/SpawnGen.Server -c Release -r "$rid" --self-contained -o "artifacts/SpawnGen-server-$rid" --nologo
done
du -sh artifacts/SpawnGen-*
