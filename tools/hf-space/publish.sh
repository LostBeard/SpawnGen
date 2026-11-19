#!/usr/bin/env bash
# Publishes a GitHub Pages build of the Studio to the Hugging Face Space LostBeard/spawngen (static SDK): the same AOT files, with the
# base href back to "/" (Pages serves the app under /SpawnGen/, the Space at its root) and this folder's README.md as the Space card.
# Needs gh (signed in to GitHub) and hf (signed in to Hugging Face with write access to the Space).
#   tools/hf-space/publish.sh [pages-run-id]     (default: the newest successful "Deploy Studio to GitHub Pages" run)
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
run="${1:-$(gh run list -R LostBeard/SpawnGen --workflow pages.yml --status success -L 1 --json databaseId -q '.[0].databaseId')}"
commit="$(gh run view "$run" -R LostBeard/SpawnGen --json headSha -q .headSha | cut -c1-7)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "Pages run $run (commit $commit) -> $work"
gh run download "$run" -R LostBeard/SpawnGen -n github-pages -D "$work/art"
mkdir -p "$work/site"
tar -xf "$work/art/artifact.tar" -C "$work/site"
cd "$work/site"
sed -i 's|<base href="/SpawnGen/" />|<base href="/" />|' index.html 404.html
grep -q '<base href="/" />' index.html || { echo "index.html has no base href to rebase"; exit 1; }
rm -f index.html.br index.html.gz 404.html.br 404.html.gz .nojekyll   # the precompressed copies still say /SpawnGen/
cp "$repo_root/tools/hf-space/README.md" README.md
hf upload LostBeard/spawngen . . --type space --delete "*" --commit-message "SpawnGen Studio $commit (GitHub Pages run $run, base href /)"
echo "https://huggingface.co/spaces/LostBeard/spawngen"
