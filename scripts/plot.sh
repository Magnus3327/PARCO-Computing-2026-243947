#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Resolve script directory (macOS-safe)
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$SCRIPT_DIR"
PY_SCRIPTS="$ROOT_DIR/scripts/plots"
JSON_FILE="$ROOT_DIR/results/distributedSPMV.json"
OUT_DIR="$ROOT_DIR/results/plots"

# -----------------------------
# Sanity checks (non sei in gita)
# -----------------------------
[[ -f "$JSON_FILE" ]] || { echo "ERRORE: $JSON_FILE non trovato"; exit 1; }
[[ -d "$PY_SCRIPTS" ]] || { echo "ERRORE: $PY_SCRIPTS non trovato"; exit 1; }

# -----------------------------
# Create output directory
# -----------------------------
mkdir -p "$OUT_DIR"

# -----------------------------
# Create venv (macOS)
# -----------------------------
python3 -m venv plot_env
source plot_env/bin/activate

# -----------------------------
# Install dependencies
# -----------------------------
pip install --upgrade pip >/dev/null
pip install matplotlib numpy pandas

# -----------------------------
# Run plots
# -----------------------------
python "$PY_SCRIPTS/strongScaling.py"   "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/weakScaling.py"     "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/strongScalingSE.py" "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/weakScalingSE.py"   "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/loadBalancing.py"   "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/memoryFootprintScaling.py" "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/structVSunstruct.py" "$JSON_FILE" "$OUT_DIR"
python "$PY_SCRIPTS/performance.py" "$JSON_FILE" "$OUT_DIR"

# -----------------------------
# Cleanup
# -----------------------------
deactivate
rm -rf plot_env

echo "Plot generati correttamente in $OUT_DIR"