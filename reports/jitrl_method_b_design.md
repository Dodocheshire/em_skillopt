# Method B: JitRL 在 EM-Benchmark 上的迁移说明

## 1. 官方 JitRL 的核心流程

官方实现（`/home/pengcheng/JitRL`）是 test-time、无梯度方法：

1. 维护跨 episode 的 `<state, action, reward>` 记忆；
2. 当前状态检索相似历史 step；
3. 用折扣 return 估计每个 action 的 advantage；
4. 对候选 action 的 logit/probability 做加性修正：

```text
corrected_logprob = normalized_prob + weighted_normalized_advantage
```

5. 选择修正后分数最高的 action；
6. episode 结束后把完整轨迹写回记忆。

官方还使用：
- OpenAI embedding + FAISS 双索引（history/state）；
- LLM step scoring / history summary；
- logprobs 或 verbalized confidence。

参考文件：

```text
JitRL/Jericho/src/jitrl_agent.py
JitRL/Jericho/src/cross_episode_memory.py
```

## 2. 迁移到 EM 的关键限制

EM 的 target 是远程 Qwen OpenAI-compatible API，当前调用只返回文本，不暴露 token logprobs，也不能直接修改 logits。

因此不能逐字复刻官方的：

```text
normalized_prob + advantage -> corrected_logprob
```

EM 的 target 输出是完整 JSON plan，而不是单个 token 的分布。

## 3. 本实现：logit-free JitRL

代码：

```text
em_skillopt/jitrl/memory.py
em_skillopt/jitrl/policy.py
```

### 记忆

每一步存：

```text
state_text
action
discounted return
success
episode_id / env / capability / step
```

持久化为 JSONL。

### 检索

用 token Jaccard 相似度检索 `top_k` 个相似历史 step，不依赖 embedding/FAISS。  
这是为了不增加额外 API 调用和依赖。

### advantage

```text
action_average_return - baseline_return
```

支持数 `support_count` 同时保留。

### 动作修正

官方修正 logit；这里修正 **parsed plan**：

1. 按 action advantage 重排 plan 中的 action；
2. 可选地把记忆中的高 advantage action 插入 plan 开头；
3. 只接受当前 action space 中存在的 action。

这不是严格的 logit 等价实现，而是 EM/API 约束下的 plan-level 近似。

## 4. 与官方的差异

| 项目 | 官方 JitRL | 本实现 |
|---|---|---|
| 策略接口 | token logits/probs | parsed action plan |
| 检索 | embedding + FAISS | token Jaccard |
| step reward | LLM step score | EM reward + terminal success |
| 记忆 | 双向量库 | JSONL |
| candidate | top-k token options | plan actions + memory action |
| 额外 LLM | summary/scoring | 无 |

## 5. 使用

```bash
cd /home/pengcheng/EM-Benchmark/em_skillopt

ENV=eb-nav \
EVAL_SETS="base common_sense complex_instruction visual_appearance long_horizon" \
WORKERS=1 \
bash scripts/run_jitrl.sh
```

或者直接：

```bash
python -m em_skillopt.cli.evaluate \
  --env eb-nav \
  --method method_b \
  --eval-set base \
  --split test \
  --parallel-workers 1 \
  --jitrl-memory runs/jitrl/eb-nav/memory.jsonl \
  --out artifacts/method_b_eb-nav/base
```

`--parallel-workers 1` 是推荐值，因为 Method B 是 online memory；多 worker 会引入记忆写入顺序的不确定性。

## 6. 后续可改进

1. 用 embedding 替换 Jaccard；
2. 让 target 一次输出 top-k actions + verbalized confidence，再做 advantage 重排；
3. 用 LLM step scorer 替代纯 terminal reward；
4. 对 plan 级修正做 validity 检查，避免重排导致不可执行序列；
5. 在 nav 上做 A/B：Method B vs baseline，固定 45 条 val / 官方 test。
