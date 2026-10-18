#!/usr/bin/env bash
# Create a virtual environment and install the core requirements.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

echo
echo "Installed. Activate the environment with:"
echo "  source .venv/bin/activate"
echo
echo "Then verify the whole pipeline in about a minute:"
echo "  python tests/smoke.py"
