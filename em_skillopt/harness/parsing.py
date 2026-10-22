"""JSON 提取 + 通用校验

通用 extract_json
high-level/nav：校验 executable_plan 里的 action_id 范围
manipulation：调用 parse_manipulation_plan，校验 7 维范围和类型
保留 EM 的 fallback 语义：空计划停止、invalid 计 format error、禁止随机动作；baseline 和 Method A 用同一策略

"""