#!/usr/bin/env bash
# Method B (JitRL logit-free EM adaptation) evaluation on official EM test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$HOME/.config/em-skillopt/client.env}"
if [[ ! -f "$ENV_FILE" && -f /home/pengcheng/SkillOpt/.env ]]; then ENV_FILE=/home/pengcheng/SkillOpt/.env; fi

PROJECT="$ROOT/em_skillopt"
PY="${PYTHON_BIN:-/opt/conda/envs/embench_nav/bin/python}"

ENV="${ENV:-eb-nav}"
OUT="${OUT:-$PROJECT/artifacts/method_b_${ENV}}"
EVAL_SETS="${EVAL_SETS:-base common_sense complex_instruction visual_appearance long_horizon}"
WORKERS="${WORKERS:-1}"
MODEL="${MODEL:-qwen3.8-27b}"
PROXY="${PROXY-}"
MEMORY="${MEMORY:-$OUT/jitrl_memory.jsonl}"

export DISPLAY="${DISPLAY:-:1}"
export PYTHONUNBUFFERED=1
export QWEN_CHAT_THINKING_MODE="${QWEN_CHAT_THINKING_MODE:-disabled}"
export PYTHONPATH="$ROOT/EmbodiedBench:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$OUT"
for EVAL_SET in $EVAL_SETS; do
  echo "===== Method B / $ENV / $EVAL_SET ====="
  "$PY" -m em_skillopt.cli.evaluate \
    --env "$ENV" \
    --method method_b \
    --eval-set "$EVAL_SET" \
    --split test \
    --parallel-workers "$WORKERS" \
    --max-tokens 8192 \
    --model "$MODEL" \
    --proxy "$PROXY" \
    --jitrl-memory "$MEMORY" \
    --jitrl-top-k "${JITRL_TOP_K:-5}" \
    --jitrl-gamma "${JITRL_GAMMA:-0.95}" \
    --jitrl-success-bonus "${JITRL_SUCCESS_BONUS:-1.0}" \
    --out "$OUT/$EVAL_SET" 2>&1 | tee "$OUT/$EVAL_SET/run.log"
done
