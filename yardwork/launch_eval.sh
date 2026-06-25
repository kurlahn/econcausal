#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-drawings/experiment/main.yaml}"

econcausal appraise --config "${CONFIG}"
econcausal chart --config "${CONFIG}"
