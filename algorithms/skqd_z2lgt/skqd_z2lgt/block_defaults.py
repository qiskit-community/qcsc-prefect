"""Shared block and runtime defaults for the SKQD Z2LGT workflow."""

from __future__ import annotations

DEFAULT_RUNTIME_NAME = "ibm-runner"
DEFAULT_OPTIONS_VARIABLE_NAME = "skqd-z2lgt-sampler-options"
DEFAULT_HPC_PROFILE_BLOCK_NAME = "hpc-miyabi-skqd-z2lgt"

DEFAULT_COMMAND_BLOCK_NAMES = {
    "dmrg": "cmd-skqd-z2lgt-dmrg",
    "preprocess": "cmd-skqd-z2lgt-preprocess",
    "train": "cmd-skqd-z2lgt-train",
    "diagonalize": "cmd-skqd-z2lgt-diagonalize",
}

DEFAULT_EXECUTION_PROFILE_BLOCK_NAMES = {
    "dmrg": "exec-skqd-z2lgt-dmrg-cpu",
    "preprocess": "exec-skqd-z2lgt-preprocess-cpu",
    "train": "exec-skqd-z2lgt-train-gpu",
    "diagonalize": "exec-skqd-z2lgt-diagonalize-gpu",
}

DEFAULT_SCRIPT_FILENAMES = {
    "dmrg": "skqd_z2lgt_dmrg.job",
    "preprocess": "skqd_z2lgt_preprocess.job",
    "train": "skqd_z2lgt_train.job",
    "diagonalize": "skqd_z2lgt_diagonalize.job",
}

DEFAULT_METRICS_ARTIFACT_KEYS = {
    "dmrg": "skqd-z2lgt-dmrg-metrics",
    "preprocess": "skqd-z2lgt-preprocess-metrics",
    "train": "skqd-z2lgt-train-metrics",
    "diagonalize": "skqd-z2lgt-diagonalize-metrics",
}

DEFAULT_COMMAND_NAMES = {
    "dmrg": "skqd-z2lgt-dmrg",
    "preprocess": "skqd-z2lgt-preprocess",
    "train": "skqd-z2lgt-train",
    "diagonalize": "skqd-z2lgt-diagonalize",
}

DEFAULT_PROFILE_NAMES = {
    "dmrg": "skqd-z2lgt-dmrg-cpu",
    "preprocess": "skqd-z2lgt-preprocess-cpu",
    "train": "skqd-z2lgt-train-gpu",
    "diagonalize": "skqd-z2lgt-diagonalize-gpu",
}
