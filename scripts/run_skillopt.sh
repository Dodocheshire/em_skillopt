#!/usr/bin/env bash
# Minimal Method A entry point.  Defaults are a small smoke run.
set -euo pipefail

ROOT=/home/pengcheng/EM-Benchmark
PROJECT="$ROOT/em_skillopt"
EMBENCH="$ROOT/EmbodiedBench"
PY="${PYTHON_BIN:-/opt/conda/envs/embench_nav/bin/python}"

MODEL="${MODEL:-qwen3.8-27b}"
PROXY="${PROXY:-http://127.0.0.1:7890}"
EPOCHS="${EPOCHS:-${STEPS:-1}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
REFLECTION_MINIBATCH_SIZE="${REFLECTION_MINIBATCH_SIZE:-8}"
REFLECTION_WORKERS="${REFLECTION_WORKERS:-0}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8}"
VAL_LIMIT="${VAL_LIMIT:-4}"
OUT="${OUT:-$PROJECT/runs/method_a/eb-nav}"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT/splits/eb-nav}"

ENV_FILE=/home/pengcheng/SkillOpt/.env
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export DISPLAY="${DISPLAY:-:1}"
export PYTHONUNBUFFERED=1
export QWEN_CHAT_THINKING_MODE="${QWEN_CHAT_THINKING_MODE:-disabled}"
export PYTHONPATH="$EMBENCH:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m em_skillopt.cli.train \
  --env eb-nav \
  --split-dir "$SPLIT_DIR" \
  --out "$OUT" \
  --model "$MODEL" \
  --proxy "$PROXY" \
  --epochs "$EPOCHS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --reflection-minibatch-size "$REFLECTION_MINIBATCH_SIZE" \
  --reflection-workers "$REFLECTION_WORKERS" \
  --parallel-workers "$PARALLEL_WORKERS" \
  --train-limit "$TRAIN_LIMIT" \
  --val-limit "$VAL_LIMIT" \
  --max-steps "${MAX_STEPS:-0}" \
  --max-planner-steps "${MAX_PLANNER_STEPS:-20}" \
  --max-tokens "${MAX_TOKENS:-8192}" \
  --max-edits "${MAX_EDITS:-3}"
