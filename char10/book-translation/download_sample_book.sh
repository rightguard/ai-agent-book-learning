#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/sample_book"
BASE="https://raw.githubusercontent.com/bojieli/ai-agent-book/main/chapter10/book-translation/sample_book"

command -v curl >/dev/null 2>&1 || { echo "[FAIL] curl 未安装" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[FAIL] python3 未安装，无法验证官方样例哈希" >&2; exit 1; }
mkdir -p "$DEST"
STAGE="$(mktemp -d "$DEST/.download.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# Git blob SHA-1 pinned to the original four files shipped in this package.
# A changed upstream file will fail closed instead of silently mutating the experiment input.
EXPECTED=(
  57379950d0cab7b7530f806e9fa6c342f2d331be
  d91ce3f7d18b018ec57ecebf8165da49baa03893
  18dc4616381c32558150430e079ba4dfba2f6c73
  2cfe64e9cb699e1ccddf07b3fff2e5224be29b0f
)

# Complete download and integrity validation BEFORE replacing any existing sample.
for i in 1 2 3 4; do
  filename="chapter${i}.md"
  echo "[Download] $filename"
  curl --fail --location --silent --show-error --retry 3 \
    --connect-timeout 20 --max-time 90 "$BASE/$filename" -o "$STAGE/$filename"
  test -s "$STAGE/$filename" || { echo "[FAIL] 下载结果为空: $filename" >&2; exit 1; }
  python3 - "$STAGE/$filename" "${EXPECTED[$((i-1))]}" <<'PY'
import hashlib
from pathlib import Path
import sys
p = Path(sys.argv[1])
b = p.read_bytes()
sha = hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()
if sha != sys.argv[2]:
    raise SystemExit(f"[FAIL] 官方样例哈希不符: {p.name}; actual={sha}")
PY
done

for i in 1 2 3 4; do
  mv -f "$STAGE/chapter${i}.md" "$DEST/chapter${i}.md"
done
printf '\n[PASS] 四篇样例已下载并验证 Git blob SHA-1：\n'
ls -lh "$DEST"/chapter{1,2,3,4}.md
