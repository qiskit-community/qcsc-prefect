#!/bin/sh
#PJM -L "node=1"
#PJM -L "rscgrp=int"
#PJM -L "elapse=3:00:00"
#PJM -o build_dice_fugaku.%j.out
#PJM -e build_dice_fugaku.%j.err
#PJM -S

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

bash ./build_dice_fugaku.sh
