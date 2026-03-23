已完成分诊，结果写入 [analysis_triage.json](/home/l00913161/projects/pta_testcase/runs/20260323T041305Z/analysis_triage.json)。共 13 条记录，结构已校验通过。

结论上，`torch.library.Library` 被归为 `PYTORCH_BUG`，`torch.fx.node.has_side_effect` 被归为 `INSUFFICIENT_COVERAGE`，其余 11 条归为 `TEST_BUG`，主要是异常类型判断过窄或把报错时机断言在了注册/构造阶段而不是实际调用阶段。