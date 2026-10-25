#!/usr/bin/env bash
# Train Method A sequentially on all four EM environments.
set -euo pipefail

ROOT=/home/pengcheng/EM-Benchmark
PROJECT="$ROOT/em_skillopt"
cd "$PROJECT"
export PROXY="${PROXY-}"

PY_NAV=/opt/conda/envs/embench_nav/bin/python
PY_ALF=/opt/conda/envs/embench/bin/python
PY_HAB=/opt/conda/envs/embench/bin/python
PY_MAN=/opt/conda/envs/embench_man/bin/python

EPOCHS="${EPOCHS:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
REFLECTION_MINIBATCH_SIZE="${REFLECTION_MINIBATCH_SIZE:-4}"
MAX_EDITS="${MAX_EDITS:-3}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs/method_a/full4_random85}"

for ENV in eb-nav eb-alf eb-hab eb-man; do
  case "$ENV" in
    eb-nav) PY="$PY_NAV"; WORKERS=8 ;;
    eb-alf) PY="$PY_ALF"; WORKERS=8 ;;
    eb-hab) PY="$PY_HAB"; WORKERS=8 ;;
    eb-man) PY="$PY_MAN"; WORKERS=1 ;;
  esac

  echo "============================================================"
  echo "START $ENV  python=$PY  workers=$WORKERS"
  echo "============================================================"

  ENV="$ENV" \
  PYTHON_BIN="$PY" \
  OUT="$OUT_ROOT/$ENV" \
  SPLIT_DIR="$PROJECT/splits/$ENV" \
  TRAIN_LIMIT=0 \
  VAL_LIMIT=0 \
  EPOCHS="$EPOCHS" \
  ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
  REFLECTION_MINIBATCH_SIZE="$REFLECTION_MINIBATCH_SIZE" \
  PARALLEL_WORKERS="$WORKERS" \
  MAX_EDITS="$MAX_EDITS" \
  GATE_METRIC=hard \
  bash scripts/run_skillopt.sh 2>&1 | tee "/tmp/em_train_${ENV}.log"
done
