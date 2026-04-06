#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

pjsub \
  -g "$GROUP" \
  -L "node=${NODE_COUNT}" \
  -L "rscgrp=${RSCGRP}" \
  -L "elapse=${ELAPSE}" \
  -x "PJM_LLIO_GFSCACHE=${GFSCACHE}" \
  --no-check-directory \
  --llio cn-read-cache=off \
  --mpi "${MPI_OPTION}" \
  --name "${JOB_NAME}" \
  "$SCRIPT_DIR/build_dice_fugaku_job.sh"
