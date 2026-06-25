#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-drawings/experiment/main.yaml}"
NPROC="${NPROC:-8}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  -m econcausal.helm.cli fit --config "${CONFIG}"
