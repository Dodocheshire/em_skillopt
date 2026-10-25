# Method A Nav 20-step Reject 实验报告

## 1. 实验设置

- 环境：`eb-nav`
- split：episode 随机 85/15，`train=255`，`val=45`
- 配置：`epochs=2`、`rollout_batch=16`、`reflection_minibatch=4`、`max_edits=3`、`workers=8`、`max_tokens=8192`
- gate：`hard`，按 val success rate 比较
- 实际结果：只完成 epoch 1 的 20 个 step，全部 reject 后已停止
- 输出：`runs/method_a/full4_random85_direct/eb-nav`

## 2. 结果

| 指标 | 数值 |
|---|---:|
| completed steps | 20 |
| accepted steps | 0 |
| rejected edits | 58 |
| current val | 22/45 = 44.4% |
| candidate val | 13.3%–42.2% |
| best_step | 0 |
| best.md | 空 |

结论：20 次 candidate 全部低于空 skill，Method A 没有产生可保留的 skill。

## 3. Prompt 分析

### 3.1 Target prompt

Nav target prompt 复用原始 EM navigation prompt：

- image first；
- system prompt 放在 user message；
- `## Task Execution Example` 与原始格式一致；
- 空 skill 时与 baseline byte-identical。

Method A 只在 system prompt 末尾追加：

```text
## Learned Skill (Method A)

The guidance below is learned strategy. The hard action-validity rules
and environment constraints in this prompt always take precedence.

<skill text>
```

问题：skill 是整个基础 prompt 之后的一大段附加指令，没有要求“最小改动”，也没有明确告诉 target 哪些旧规则应被覆盖。长规则容易与原始 few-shot/策略文本叠加，而不是精准修正一个失败模式。

### 3.2 Reflect prompt

`reflect.py` 对 failure 和 success 分开调用，输入包含：

- instruction、success/reward、failure_reason；
- full target system/user prompt；
- full plans 和 full trajectory（默认不截断）。

system prompt 关键要求：

```text
Edits must be short, actionable, reusable rules.
Do not mention episode IDs.
Return at most {max_edits} edits.
```

问题：

1. 只有 “short” 这一句，没有字符数/词数上限；
2. 没有要求必须由至少 2 条 episode 支持；
3. 没有要求只修正一个具体失败模式；
4. 没有禁止重写整体策略；
5. failure 和 success 分开调用，最后直接合并，缺少“哪个失败模式最值得修”的约束。

### 3.3 Merge prompt

`aggregate.py` 分层合并：

1. failure patches 先合并；
2. success patches 再合并；
3. 最后 failure + success 合并。

关键指令：

```text
Remove duplicates and contradictory rules.
Prefer failure-driven edits when two edits conflict.
Preserve support counts when possible.
```

问题：明确偏向 failure-driven edits，且没有限制合并后的总长度；它倾向于把多个失败规则“整理成完整策略”，而不是保留最小修正。

### 3.4 Clip prompt

`clip.py` 让 optimizer 按“expected benefit”排序，保留 top `max_edits`。

问题：

- 没有要求优先保留短、局部、support 高的 edit；
- 没有要求不同 edit 之间互不重复；
- 最终只按排名截断，不能阻止 3 条大规则一起进入 candidate。

## 4. 生成 edit 的特征

20 个 candidate 的统计：

| 指标 | 数值 |
|---|---:|
| edit 形式 | 全部 `append` |
| 每 candidate edit 数 | 3（step18 为 1） |
| 长度 | 859–1549 chars，平均 1195 |
| 词数 | 135–255，平均 191 |
| support_count | 0–6，平均 4.5 |
| 来源 | 16/20 全部 failure；2/20 混合；1/20 failure+success；1/20 success-only |

高频内容：

```text
target 100 次
lateral 67
forward 56
obstacle 53
rotate 31
backward 22
collision 20
2-3 actions 13
distractor 13
```

典型 candidate 包含：

- 严格识别 target，排除 distractor；
- 被挡住时优先 lateral，后退 1–2 步；
- 不要原地旋转；
- plan 限制为 2–3 actions；
- 频繁 re-evaluate。

这些规则本身并非错误，但它们是**整套导航策略的复述**，不是针对某个具体失败轨迹的最小修正。

## 5. Candidate 为什么失败

### 5.1 一次性改动过大

3 条 append 平均约 1200 字符，相当于在原始 prompt 后追加一份新的策略说明。任何一条规则过强，都可能改变原有行为。

### 5.2 Failure-only 偏置

16/20 candidate 完全来自 failure patches；merge prompt 又明确“failure 优先”。这会把偶然失败过度泛化成全局规则。

### 5.3 Support 太低

平均 support 4.5，单个 edit 常见 support 1–2。minibatch 只有 4 个 episode，失败样本更少，规则很容易只适配个别场景。

### 5.4 与基础 prompt 重复/冲突

原始 nav prompt 已有动作空间、示例和 1–2 / 5–6 actions 的要求。candidate 再追加 “2–3 actions”“不要旋转”“先 lateral”等规则，可能覆盖或冲突。

### 5.5 最短 candidate 最接近通过

step18 只有 1 条 success edit、294 chars，candidate val = 42.2%，是 20 次中最接近 current 44.4% 的一次。  
这支持“少改动”方向，但也说明单条 success edit 仍可能不够。

## 6. 结论

这次全部 reject 不是 runner 或 split 的机械故障，而是 edit 生成/合并链路的问题：

```text
3 条大段 append
+ failure-only 偏置
+ support 低
+ merge 偏重失败规则
+ clip 只看排名，不看最小性
= candidate 系统性弱于空 skill
```

## 7. 建议

1. Reflect 强制只生成 **1 条** edit，长度上限 200–300 chars；
2. 要求该 edit 必须由至少 2 条 episode 支持，并写明具体失败模式；
3. 禁止“重写整体策略”，只允许局部 append/insert；
4. Merge 不再默认 failure 优先，改为“保留 support 高且互不重复的最小集合”；
5. Clip 增加长度惩罚，优先短 edit；
6. 先用 `MAX_EDITS=1` 的 10-step probe 验证；若仍全 reject，再修改 reflect prompt，而不是继续增加训练步数。

## 8. 后续 MAX_EDITS=1 probe 异常

`MAX_EDITS=1` probe 启动后，step 1 出现 8 个 Thor/Unity 崩溃：

```text
Unity process has exited ... returncode=-11
```

step 1 未完成，没有产生有效 step 结果，已停止该 probe。该崩溃是运行环境问题，不能用它判断单条 edit 的效果；需要先降低并行数或修复 Unity/Vulkan 后再重跑。
