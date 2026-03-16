#!/usr/bin/env bash
set -euo pipefail

# Always operate in this script directory to avoid accidental cleanup in cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_SBD_DIR="${SCRIPT_DIR}/../../gb_sqd/gb_demo_2026/deps/sbd"
SBD_DIR="${SBD_DIR:-$DEFAULT_SBD_DIR}"
CCCOM="${CCCOM:-mpic++}"
CCFLAGS="${CCFLAGS:--std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST}"
SYSLIB="${SYSLIB:--lblas -llapack}"

if ! command -v "$CCCOM" >/dev/null 2>&1; then
    echo "Compiler '$CCCOM' not found in PATH. Load the Miyabi-G compiler/MPI environment first." >&2
    exit 1
fi

if [ ! -d "$SBD_DIR" ]; then
    echo "SBD include directory not found: $SBD_DIR" >&2
    echo "Set SBD_DIR to the SBD source tree you want to use." >&2
    exit 1
fi

# Clean previous build
rm -f "$SCRIPT_DIR"/*.o "$SCRIPT_DIR"/diag-gpu

# Compile and link
echo "$CCCOM $CCFLAGS -c main.cc -o main.o -I$SBD_DIR/include"
$CCCOM $CCFLAGS -c main.cc -o main.o -I"$SBD_DIR/include"
echo "$CCCOM $CCFLAGS $SYSLIB -o diag-gpu main.o"
$CCCOM $CCFLAGS $SYSLIB -o diag-gpu main.o

echo "Build completed: $SCRIPT_DIR/diag-gpu"
