# 新服务器 / 新 DSH Workspace 交接（给 Mac 本机 Agent）

## 0. 当前目标

在新服务器 `/home/delci/EM-Benchmark` 上继续：

```text
baseline
Method A = SkillOpt learnable skill
Method B = JitRL-style test-time memory / advantage
```

target 和 optimizer 都只用：

```text
qwen3.8-27b
```

当前不在新服务器上启动大规模 baseline；先 smoke，再跑单 worker 的 nav probe。

---

## 1. 新服务器访问

Mac 本机 SSH 配置应包含：

```sshconfig
Host em-new
    HostName 10.162.141.11
    User delci
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ConnectTimeout 20
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

验证：

```bash
ssh em-new
hostname
# cheshirecat
whoami
# delci
```

注意：Windows 侧用户态代理监听 22，转发到 WSL 内 sshd 的 2222。  
**Mac 上不要直接连 2222**，用 `em-new`（22）即可。

当前已能登录；仓库在：

```text
/home/delci/EM-Benchmark/em_skillopt
```

---

## 2. 新服务器实际环境（已由 README_WSL.md 确认）

```text
Ubuntu 26.04.1 LTS, WSL2
Miniforge: /home/delci/miniforge3
/opt/conda -> /home/delci/miniforge3
```

三个 conda env，均为 Python 3.9.23：

```text
embench_nav:
  ai2thor 5.0.0
  torch 2.4.0+cpu
  jsonschema 4.25.1

embench:
  ai2thor 2.1.0
  habitat / habitat-sim 0.3.0
  jsonschema 4.25.1

embench_man:
  PyRep + CoppeliaSim 4.1
  ultralytics, open3d
  jsonschema 4.25.1
```

显示/Vulkan：

```text
Xvfb :1 已运行
Mesa Dozen Vulkan driver
Vulkan device:
  Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)
  driver Dozen
```

CloudRendering 和 nav smoke 已通过：

```text
success 1/1
format_error 0
elapsed 30.9s
```

AI2-THOR release：

```text
~/.ai2thor/releases/thor-CloudRendering-f082...
~/.ai2thor/releases/thor-Linux64-f082...
```

Qwen 配置已存在：

```text
~/.config/em-skillopt/client.env
```

数据大致：

```text
ALFRED       ~539 MB
Habitat      ~1.5 GB
Manipulation ~1.9 GB
AI2-THOR     ~2.1 GB
```

---

## 3. 代码版本和路径

当前仓库：

```text
https://github.com/Dodocheshire/em_skillopt.git
```

旧服务器已有 commit：

```text
8d26386 Add JitRL Method B and simplify training splits
a47a674 Add new server handoff notes
```

本次又加入了脚本路径便携化（`SCRIPT_DIR` / `EB_ROOT` / `ENV_FILE`）和本文件。  
新服务器 Agent 第一步应：

```bash
cd /home/delci/EM-Benchmark/em_skillopt
git pull --ff-only
git log -1 --oneline
```

如果 GitHub 访问不了，使用旧服务器或本机已有 checkout 同步。

EB 仓库：

```text
https://github.com/Dodocheshire/EB-Benchmark.git
当前参考: 93e53e2
```

---

## 4. 运行前设置

脚本现在会自动从自身位置推导 `ROOT`：

```bash
export ROOT=/home/delci/EM-Benchmark
export PYTHONPATH="$ROOT/EmbodiedBench:$ROOT/em_skillopt"
export DISPLAY=:1
```

Qwen 配置从：

```text
~/.config/em-skillopt/client.env
```

读取。不要提交 API key。

如果代理正常，脚本默认 `PROXY=http://127.0.0.1:7890`；  
如果代理异常，设置：

```bash
PROXY=
```

或显式传空 `--proxy ""`。

---

## 5. 第一次在新服务器上做的事

### 5.1 读 README 和状态

```bash
ssh em-new
cd /home/delci/EM-Benchmark

sed -n '1,260p' README_WSL.md
sed -n '1,220p' em_skillopt/reports/server_setup.md
```

### 5.2 更新代码

```bash
cd /home/delci/EM-Benchmark/em_skillopt
git pull --ff-only
git status --short
```

### 5.3 确认显示

```bash
xdpyinfo -display :1 >/dev/null && echo DISPLAY_OK
vulkaninfo --summary | grep -E 'deviceName|driverName'
```

Xvfb 已由 systemd 启动；如果不在：

```bash
sudo systemctl status xvfb.service
```

不要随意改用 `start_headless.sh`；WSL 没有 NVIDIA PCI 设备，脚本会报 `no nvidia cards found`。

### 5.4 单条 nav smoke

```bash
source ~/.config/em-skillopt/client.env
export ROOT=/home/delci/EM-Benchmark
export PYTHONPATH="$ROOT/EmbodiedBench:$ROOT/em_skillopt"
export DISPLAY=:1

cd "$ROOT/em_skillopt"

/opt/conda/envs/embench_nav/bin/python -m em_skillopt.cli.evaluate \
  --env eb-nav \
  --method baseline \
  --eval-set base \
  --split test \
  --limit 1 \
  --parallel-workers 1 \
  --max-tokens 8192 \
  --proxy "" \
  --out "$ROOT/runs/smoke_nav_delci"
```

预期参考：

```text
success_rate ≈ 1.0
format_error_rate = 0
```

### 5.5 单元测试

```bash
cd "$ROOT/em_skillopt"
PYTHONPATH="$ROOT/EmbodiedBench:$ROOT/em_skillopt" \
  /opt/conda/envs/embench_nav/bin/python -m unittest discover -s tests -q
```

---

## 6. 当前实验上下文

旧服务器上的 Method A probe：

```text
runs/method_a/eb-nav_maxedits1_probe_w1
env: eb-nav
MAX_EDITS=1
MAX_TRAIN_STEPS=10
episode workers=1
```

旧服务器已完成：

```text
step 1: reject
  current 37.8% -> candidate 35.6%

step 2: accept_new_best
  current 37.8% -> candidate 46.7%
  best_step=2
  best skill=444 chars

step 3: reject
step 4: running when this handoff was written
```

结论：

```text
MAX_EDITS=1 能产生非空 best
之前 3 条大段 append 全部 reject，原因是改动过大
```

注意：step 2 接受的规则包含：

```text
If target is on a surface and within 1-2 steps but cannot move closer,
consider the task complete.
```

这可能在官方 test 上过拟合，必须用官方 test 验证。

Method B：

```text
em_skillopt/jitrl/memory.py
em_skillopt/jitrl/policy.py
em_skillopt/cli/evaluate.py --method method_b
scripts/run_jitrl.sh
```

Method B 是 **logit-free** 的 plan-level 近似；官方 JitRL 需要 token logprobs/logits，Qwen API 不提供。

---

## 7. 新服务器上的建议实验顺序

### A. 先跑 nav Method A 单 edit probe

```bash
cd /home/delci/EM-Benchmark/em_skillopt

ENV=eb-nav \
PYTHON_BIN=/opt/conda/envs/embench_nav/bin/python \
OUT=runs/method_a/eb-nav_maxedits1_probe \
SPLIT_DIR=splits/eb-nav \
TRAIN_LIMIT=0 VAL_LIMIT=0 EPOCHS=1 \
ROLLOUT_BATCH_SIZE=16 REFLECTION_MINIBATCH_SIZE=4 \
PARALLEL_WORKERS=1 MAX_EDITS=1 MAX_TRAIN_STEPS=10 \
GATE_METRIC=hard PROXY= \
bash scripts/run_skillopt.sh
```

单 worker 在 nav 上一个 step 很慢（含 45 条 current + 45 条 candidate val）。  
如果确认 GPU/CloudRendering 稳定，可再考虑提高到 2；不要直接 8。

### B. 如果出现 accept

先不要直接扩到其他 env；用官方 nav test 验证 accepted skill：

```bash
for EVAL_SET in base common_sense complex_instruction visual_appearance long_horizon; do
  /opt/conda/envs/embench_nav/bin/python -m em_skillopt.cli.evaluate \
    --env eb-nav \
    --method method_a \
    --skill-file runs/method_a/eb-nav_maxedits1_probe/skill/best.md \
    --eval-set "$EVAL_SET" \
    --split test \
    --parallel-workers 1 \
    --max-tokens 8192 \
    --proxy "" \
    --out "artifacts/method_a_official/$EVAL_SET"
done
```

### C. Method B

```bash
ENV=eb-nav \
EVAL_SETS="base common_sense complex_instruction visual_appearance long_horizon" \
WORKERS=1 \
bash scripts/run_jitrl.sh
```

先做 base 20 条对照，再扩全量。

---

## 8. 注意事项

- 不要在新服务器上启动大型 baseline；先 smoke。
- 不要把 API key 写入 git。
- 不要同时跑 nav + alf/hab/man 的大量 worker；WSL/Dozen 比裸机更容易受并发影响。
- `embench` 的旧 ai2thor 2.1.0 可能会删除 CloudRendering；如果 nav 反复下载，检查 `prune_releases()`，参见 `reports/server_setup.md`。
- `run_baseline.sh` 默认 8192 tokens；man 必须 `MAN_PARALLEL_WORKERS=1`。
- 旧服务器 probe 仍在运行；新服务器不要重复启动同一输出目录。

---

## 9. 新 DSH 会话第一条消息建议

```text
工作目录：/home/delci/EM-Benchmark/em_skillopt
代码版本：git pull 后最新 main
先阅读 README_WSL.md 和 reports/server_setup.md。
不要启动四环境大规模训练；先做单条 nav smoke。
模型只用 qwen3.8-27b，不要提交 API key。
接着验证 Method A 的 MAX_EDITS=1 单 worker probe，再考虑 Method B。
```
