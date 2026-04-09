#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x \
  "$SCRIPT_DIR/build_dice_fugaku.sh" \
  "$SCRIPT_DIR/build_dice_fugaku_job.sh"

GROUP="${FUGAKU_GROUP:-${PROJECT:-}}"
if [ -z "${GROUP:-}" ]; then
    echo "Set FUGAKU_GROUP (or PROJECT) before submitting the Fugaku build job." >&2
    exit 1
fi

RSCGRP="${FUGAKU_BUILD_RSCGRP:-int}"
NODE_COUNT="${FUGAKU_BUILD_NODE_COUNT:-1}"
ELAPSE="${FUGAKU_BUILD_ELAPSE:-3:00:00}"
GFSCACHE="${FUGAKU_BUILD_GFSCACHE:-/vol0004}"
MPI_OPTION="${FUGAKU_BUILD_MPI_OPTION:-max-proc-per-node=1}"
JOB_NAME="${FUGAKU_BUILD_JOB_NAME:-dice-build}"
WAIT_TIME="${FUGAKU_BUILD_WAIT_TIME:-600}"
SUBMIT_MODE="${FUGAKU_BUILD_SUBMIT_MODE:-auto}"
GENERATED_JOB_SCRIPT="$SCRIPT_DIR/.build_dice_fugaku_job.generated.sh"

if [ "$SUBMIT_MODE" = "auto" ]; then
    if [ "$RSCGRP" = "int" ]; then
        SUBMIT_MODE="interact"
    else
        SUBMIT_MODE="batch"
    fi
fi

if [ "$SUBMIT_MODE" = "interact" ]; then
    pjsub --interact \
      -g "$GROUP" \
      -L "node=${NODE_COUNT}" \
      -L "rscgrp=${RSCGRP}" \
      -L "elapse=${ELAPSE}" \
      --sparam "wait-time=${WAIT_TIME}" \
      -x "PJM_LLIO_GFSCACHE=${GFSCACHE}" \
      --no-check-directory \
      --llio cn-read-cache=off \
      --mpi "${MPI_OPTION}" <<EOF
cd "$SCRIPT_DIR"
bash ./build_dice_fugaku.sh
exit
EOF
    exit 0
fi

cat > "$GENERATED_JOB_SCRIPT" <<EOF
#!/bin/sh
#PJM -L "node=${NODE_COUNT}"
#PJM -L "rscgrp=${RSCGRP}"
#PJM -L "elapse=${ELAPSE}"
#PJM -x PJM_LLIO_GFSCACHE=${GFSCACHE}
#PJM --mpi "${MPI_OPTION}"
#PJM --name ${JOB_NAME}
#PJM -o build_dice_fugaku.%j.out
#PJM -e build_dice_fugaku.%j.err
#PJM -S

set -eu

SCRIPT_DIR="\$(CDPATH= cd -- \"\$(dirname -- \"\$0\")\" && pwd)"
cd "$SCRIPT_DIR"

bash ./build_dice_fugaku.sh
EOF
chmod +x "$GENERATED_JOB_SCRIPT"

pjsub \
  -g "$GROUP" \
  --no-check-directory \
  --llio cn-read-cache=off \
  "$GENERATED_JOB_SCRIPT"
