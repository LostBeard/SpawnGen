#!/usr/bin/env bash
# One copy of each file on the site: the Studio publishes uncompressed files only (CompressionEnabled=false in its csproj; .NET's default
# would add a .gz AND a .br copy of every file, tripling the upload, and nothing ever read the .br ones). Where a host needs a compressed
# file, this script swaps the file for its .gz copy and lists it in gzboot.json, which index.html reads before it starts the runtime: a
# listed file is fetched as <name>.gz and inflated in the browser (DecompressionStream).
#   tools/gzboot.sh <wwwroot> all           every _framework file except .js / .json: for a host that sends files uncompressed
#                                            (the Hugging Face static Space: 91 MB of runtime instead of 24 MB otherwise)
#   tools/gzboot.sh <wwwroot> over <MB>     only files larger than <MB>: for a host with a file size limit that compresses on its own
#                                            (GitHub Pages: 100 MB per file; it gzips what it serves)
set -euo pipefail
root="$1"; mode="$2"; limit="${3:-95}"
cd "$root"
names=()
while IFS= read -r -d '' f; do
  case "$mode" in
    all)  case "$f" in *.js|*.json|*.gz) continue;; esac ;;
    over) [ "$(stat -c %s "$f")" -gt $((limit * 1024 * 1024)) ] || continue ;;
    *) echo "mode must be all or over"; exit 2 ;;
  esac
  gzip -9 -n -f "$f"            # replaces the file with <file>.gz
  names+=("$(basename "$f")")
done < <(find _framework -maxdepth 1 -type f -print0)
if [ "$mode" = all ]; then
  printf '{"all":true,"files":[]}\n' > gzboot.json
else
  list=""; for n in "${names[@]:-}"; do [ -n "$n" ] && list="$list\"$n\","; done
  printf '{"all":false,"files":[%s]}\n' "${list%,}" > gzboot.json
fi
echo "gzboot ($mode): ${#names[@]} file(s) kept as .gz only; $(cat gzboot.json | cut -c1-200)"
