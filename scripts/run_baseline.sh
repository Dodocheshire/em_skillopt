#!/usr/bin/env bash
# Run the no-skill EM-Benchmark baseline for all four environments and all
# capability subsets.
#
# Usage:
#   bash scripts/run_baseline.sh
#   LIMIT=2 bash scripts/run_baseline.sh          # smoke: 2 episodes per subset
#   MODEL=Qwen3.6-35B-A3B bash scripts/run_baseline.sh
#   MAN_NO_DETECTION=1 bash scripts/run_baseline.sh
#
# Environment overrides:
#   MODEL, PROXY, LIMIT, MAX_STEPS, MAX_PLANNER_STEPS, MAX_TOKENS, OUT_ROOT
set -uo pipefail

ROOT=/home/pengcheng/EM-Benchmark
EMBENCH="$ROOT/EmbodiedBench"
PROJECT="$ROOT/em_skillopt"
CONDA_ROOT=/opt/conda/envs

PY_NAV="$CONDA_ROOT/embench_nav/bin/python"
PY_ALF="$CONDA_ROOT/embench/bin/python"
PY_HAB="$CONDA_ROOT/embench/bin/python"
PY_MAN="$CONDA_ROOT/embench_man/bin/python"

MODEL="${MODEL:-qwen3.8-27b}"
PROXY="${PROXY:-http://127.0.0.1:7890}"
LIMIT="${LIMIT:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_PLANNER_STEPS="${MAX_PLANNER_STEPS:-20}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/artifacts/baseline}"

export DISPLAY="${DISPLAY:-:1}"
export PYTHONUNBUFFERED=1
export QWEN_CHAT_THINKING_MODE="${QWEN_CHAT_THINKING_MODE:-disabled}"

# CoppeliaSim / manipulation environment.
export COPPELIASIM_ROOT="$ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export EB_DETECTION_DEVICE="${EB_DETECTION_DEVICE:-cpu}"

ENV_FILE=/home/pengcheng/SkillOpt/.env
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export PYTHONPATH="$EMBENCH:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

FAILED=()

run_one() {
    local env_name="$1"
    local python_bin="$2"
    local eval_set="$3"
    local out="$OUT_ROOT/$env_name/$eval_set"
    local -a extra=()

    if [[ "$env_name" == "eb-man" && "${MAN_NO_DETECTION:-0}" == "1" ]]; then
        extra+=(--no-detection)
    fi

    mkdir -p "$out"
    printf '\n===== %s / %s =====\n' "$env_name" "$eval_set"
    printf 'model=%s limit=%s out=%s\n' "$MODEL" "$LIMIT" "$out"

    if ! "$python_bin" -m em_skillopt.cli.evaluate \
        --env "$env_name" \
        --method baseline \
        --eval-set "$eval_set" \
        --split test \
        --limit "$LIMIT" \
        --max-steps "$MAX_STEPS" \
        --max-planner-steps "$MAX_PLANNER_STEPS" \
        --max-tokens "$MAX_TOKENS" \
        --model "$MODEL" \
        --proxy "$PROXY" \
        --out "$out" \
        "${extra[@]}" 2>&1 | tee "$out/run.log"
    then
        echo "FAILED: $env_name / $eval_set" >&2
        FAILED+=("$env_name/$eval_set")
    fi
}

run_environment() {
    local env_name="$1"
    local python_bin="$2"
    shift 2
    local eval_set
    for eval_set in "$@"; do
        run_one "$env_name" "$python_bin" "$eval_set"
    done
}

# Capability subset names must match each environment's ValidEvalSets.
run_environment eb-nav "$PY_NAV" \
    base common_sense complex_instruction visual_appearance long_horizon

run_environment eb-alf "$PY_ALF" \
    base common_sense complex_instruction spatial visual_appearance long_horizon

run_environment eb-hab "$PY_HAB" \
    base common_sense complex_instruction spatial_relationship visual_appearance long_horizon

run_environment eb-man "$PY_MAN" \
    base common_sense complex spatial visual

echo
echo "============================================================"
if (( ${#FAILED[@]} == 0 )); then
    echo "Baseline finished: no failed jobs"
else
    echo "Baseline finished with failures:"
    printf '  %s\n' "${FAILED[@]}"
fi
echo "Results: $OUT_ROOT"
echo "============================================================"

(( ${#FAILED[@]} == 0 ))
