# 新服务器 / 新 DSH Workspace 交接上下文

## 1. 新服务器

```text
SSH:
  ssh -i ~/.ssh/id_ed25519 delci@10.162.141.11

已在 ~/.ssh/config 添加：
  Host em-new
      HostName 10.162.141.11
      User delci
      Port 22
      IdentityFile ~/.ssh/id_ed25519

  Host em-new-ts        # Tailscale fallback
      HostName 100.109.42.17
      User delci
      Port 22
      IdentityFile ~/.ssh/id_ed25519
```

预期工作目录：

```text
/home/delci/EM-Benchmark
```

用户说环境基本安装好了，并要求先阅读：

```text
/home/delci/EM-Benchmark/README_WSL.md
```

当前从这台 Mac 访问：

```text
10.162.141.11:22   timeout
100.109.42.17:22   timeout
```

即新服务器目前从本机不可达（WSL/Tailscale/Windows host 可能未启动或网络不同）。等服务器在线后，先读 README_WSL.md，再继续下面的部署/实验。

---

## 2. 当前代码版本

```text
em_skillopt:  https://github.com/Dodocheshire/em_skillopt.git
              8d26386 Add JitRL Method B and simplify training splits

EB-Benchmark: https://github.com/Dodocheshire/EB-Benchmark.git
              93e53e2
```

新服务器 clone 后应检查：

```bash
cd /home/delci/EM-Benchmark
git -C em_skillopt log -1 --oneline
git -C EmbodiedBench log -1 --oneline
```

---

## 3. 项目目标

把 SkillOpt / JitRL 适配到 EM-Benchmark，同时保持 EM 原始 harness。

方法：

```text
baseline:   不注入 task skill
Method A:   SkillOpt learnable skill
Method B:   JitRL-style test-time memory / advantage
target/optimizer 均使用 qwen3.8-27b
```

当前评测协议：

```text
nav 训练/验证：所有 nav episode 合并，随机 85/15
  train 255
  val 45
最终：官方 EM test（baseline 与 Method A best skill 都在官方 test 上评）
```

注意：简单随机 split 会跨 scene 泄漏，但用于同一套方法比较；官方 test 作为最终指标。

---

## 4. 旧服务器当前实验状态

旧服务器：

```text
Host: ZJU-molink
/home/pengcheng/EM-Benchmark/em_skillopt
```

当前 probe：

```text
tmux: em-train-probe
OUT: runs/method_a/eb-nav_maxedits1_probe_w1
env: eb-nav
epochs: 1
MAX_EDITS: 1
MAX_TRAIN_STEPS: 10
episode workers: 1
proxy: direct Qwen
```

已完成的 3 个 step：

```text
step 1: reject
  current 37.8% -> candidate 35.6%

step 2: accept_new_best
  current 37.8% -> candidate 46.7%
  best_step = 2
  best skill = 444 chars

step 3: reject
  best 仍为 step 2
```

当前 step 4 正在跑；没有 Unity/Thor/API 错误。

这个结果说明：

```text
MAX_EDITS=1 能产生可接受 edit
之前 3 条大段 append 全部 reject，主要是改动过大
```

但 step 2 接受的规则包含：

```text
If target is on a surface and within 1-2 steps but cannot move closer,
consider the task complete.
```

这可能是 val 过拟合，后续要在官方 test 验证。

旧的四环境队列 `em-train-all` 已停止；旧 full4 run 是 20 步全 reject，保留在：

```text
runs/method_a/full4_random85_direct/eb-nav
```

---

## 5. 新服务器需要做的事

### 5.1 读取 WSL README

```bash
ssh em-new
cd /home/delci/EM-Benchmark
sed -n '1,260p' README_WSL.md
```

重点确认：

- conda 安装位置
- 三个环境是否已建好
- Xorg / DISPLAY
- 数据集位置
- 是否有 GPU / 容器外占用
- 代码路径

### 5.2 代码

如果新服务器还没有 `em_skillopt`：

```bash
cd /home/delci/EM-Benchmark
git clone https://github.com/Dodocheshire/em_skillopt.git em_skillopt
```

如果已有，更新：

```bash
cd /home/delci/EM-Benchmark/em_skillopt
git pull --ff-only
```

### 5.3 路径修改

脚本里默认仍是：

```text
/home/pengcheng/EM-Benchmark
/home/pengcheng/SkillOpt/.env
```

新服务器是 `delci`，需要修改：

```text
scripts/run_baseline.sh
scripts/run_skillopt.sh
scripts/run_train_all.sh
scripts/run_jitrl.sh
```

或设置环境变量/建立同路径。

必须设置：

```bash
export ROOT=/home/delci/EM-Benchmark
export PYTHONPATH="$ROOT/EmbodiedBench:$ROOT/em_skillopt"
export DISPLAY=:1
```

Qwen API：

```bash
export QWEN_CHAT_BASE_URL="..."
export QWEN_CHAT_API_KEY="..."
export QWEN_CHAT_MODEL="qwen3.8-27b"
export QWEN_CHAT_THINKING_MODE=disabled
```

不要把旧服务器的 key 写进 git。

### 5.4 Split

Git 跟踪：

```text
splits/eb-nav/
```

以下被忽略，需要从旧服务器 rsync 或重新生成：

```text
splits/eb-alf/
splits/eb-hab/
splits/eb-man/
```

旧服务器路径：

```text
/home/pengcheng/EM-Benchmark/em_skillopt/splits/
```

### 5.5 数据集

如果新服务器还没有完整数据：

```bash
cd /home/delci/EM-Benchmark/EmbodiedBench
bash scripts/download_datasets.sh --dataset all
```

Navigation JSON 随仓库提供；AI2-THOR 首次启动会下载到 `~/.ai2thor`。

已知 ai2thor 2.1.0 的 `prune_releases()` 可能删除 nav 的 CloudRendering；参见：

```text
EM-Benchmark/em_skillopt/reports/server_setup.md
```

### 5.6 快速验证

```bash
xdpyinfo -display :1

cd /home/delci/EM-Benchmark/em_skillopt
PYTHONPATH="$ROOT/EmbodiedBench:$ROOT/em_skillopt" \
  /path/to/embench_nav/bin/python -m em_skillopt.cli.evaluate \
  --env eb-nav --method baseline --eval-set base --split test \
  --limit 1 --parallel-workers 1 --max-tokens 8192 \
  --proxy "" --out /tmp/em_smoke_nav
```

---

## 6. 建议的新实验顺序

1. 新服务器单条 nav smoke；
2. 跑 `MAX_EDITS=1` 的 nav probe（1 epoch，10 step）；
3. 如果 accept，继续官方 nav test；
4. 再跑 Method B JitRL；
5. 最后再回到四环境 baseline / Method A。

不要在 GPU 已被容器外用户占用时提高 worker 数。

---

## 7. 重要已知问题

- 旧服务器 GPU 有容器外用户，约 30 GB 被占；不能 `nvidia-smi --gpu-reset`，不要 reboot。
- 单 worker 很慢：一个 step 包含 16 rollout + 45 current + 45 candidate val。
- 旧 `embench` ai2thor 2.1.0 会误删 CloudRendering；需要 patch 或独立 cache。
- `run_baseline.sh` 的 `MAN_PARALLEL_WORKERS=1` 是必须的；CoppeliaSim 不能多线程。
- Xorg 需要手动启动；不要在没有确认时重启整个显示服务。
- Method B 是 logit-free 的 plan-level 近似，不是官方 JitRL 的 token-logit 复刻。
- `reports/method_a_reject_report.md` 记录了 20 步 3-edit 全 reject 的分析。
- `reports/jitrl_method_b_design.md` 记录了 Method B 与官方的差异。

---

## 8. 建议给新 DSH 会话的初始指令

```text
工作目录：/home/delci/EM-Benchmark/em_skillopt
代码版本：8d26386
先阅读 README_WSL.md 和 reports/server_setup.md。
不要启动 baseline/Method A 大规模训练；先做单条 nav smoke。
所有模型只用 qwen3.8-27b；不要提交 API key。
当前重点是验证 MAX_EDITS=1 的 Method A probe，以及 JitRL Method B。
```
