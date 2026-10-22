"""EM trajectory -> SkillOpt patch
每个 episode 压缩成 EpisodeDigest
按 failure/success 分组
每个 reflection minibatch 放 4–8 个 episode digest
失败类型用 EM 自己的分类：
invalid_action
empty_plan
json_error
wrong_object
missing_precondition
loop
max_steps
termination_error
spatial_error

digest 里只放：

instruction、env、capability
success、reward、env steps、planner steps
invalid action 次数、format error 次数
关键 planner 轮次：首轮、invalid 前后、最后一轮
每轮的 reasoning_and_reflection、language_plan、executed actions、env feedback
不放图片；optimizer 是文本模型，图片由 target 自己描述

然后复用 SkillOpt 的:
SkillOpt 模块	用途
gradient/aggregate.py	合并 failure/success patches
optimizer/clip.py	rank + top-L 选择
optimizer/skill.py	append/insert/replace/delete + protected region
evaluation/gate.py	candidate vs current on selection
optimizer/slow_update.py	epoch 末纵向对比
engine/trainer.py	训练循环参考；不要整段照搬
"""
