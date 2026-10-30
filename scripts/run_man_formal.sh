#!/usr/bin/env bash
set -eo pipefail
source /data/EM-Benchmark/env_man.sh
set -u
CODE=/data/EM-Benchmark/man_formal_ed1bc9e
OUT=/data/EM-Benchmark/runs/method_a/man_m3
export PYTHONPATH="$CODE:$PYTHONPATH"
export QWEN_CHAT_MAX_CONCURRENCY=2
# Episode spatial:0041 returned valid responses after 161/171 seconds; the
# former 120-second read timeout discarded them before receipt. This changes
# transport patience only, not prompts, model or generation token budgets.
export QWEN_CHAT_TIMEOUT_SECONDS=600
export QWEN_CHAT_HARD_TIMEOUT_S=660
cd /data/EM-Benchmark/EmbodiedBench
mkdir -p "$OUT" /data/EM-Benchmark/logs
# No automatic restart: partial skill/gate state must be inspected before resume.
"$PY" -u -m em_skillopt.cli.train \
 --env eb-man --split-dir /data/EM-Benchmark/em_skillopt/splits/eb-man \
 --out "$OUT" --model qwen3.8-27b --proxy "" \
 --epochs 2 --rollout-batch-size 48 --reflection-minibatch-size 4 \
 --reflection-workers 0 --parallel-workers 2 --n-shots 4 \
 --train-limit 0 --val-limit 0 --max-train-steps 0 --skip-steps 0 \
 --max-steps 0 --max-planner-steps 20 --max-tokens 8192 --temperature 0 \
 --max-edits 1 --gate-metric hard --gate-mixed-weight 0.5 \
 > /data/EM-Benchmark/logs/train_man_m3.log 2>&1
"$PY" -u "$CODE/finish_man_verified.py"
