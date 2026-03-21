# Failure Taxonomy

流水线会把每个 API 的失败归到统一分类，用于：

- `analysis_triage.json`
- `analysis_summary.md`
- `results.json`
- `results.csv`
- `summary.md`

这个分类的目标不是“证明根因”，而是把默认动作收敛成两类：

1. `TEST_BUG`
   可以在当前批次内继续自动修复
2. 其他分类
   先稳定分类和报告，不默认乱改

## Categories

- `NONE`
  当前最终状态已通过，没有待处理失败。
- `TEST_BUG`
  问题主要在 `test/api_test/` 下的测试代码，例如参数构造错误、断言错误、异常预期错误、收集失败、文件未生成。
- `ENVIRONMENT_MISSING`
  环境缺少 `torch_npu`、NPU 不可用、基础依赖未满足，或当前运行条件不成立。
- `UNSUPPORTED_ON_NPU`
  当前 NPU 后端、当前构建、当前 dispatch/layout 组合不支持该 API 路径。
- `PYTORCH_BUG`
  证据更偏向 `pytorch/` 内部实现问题。
- `TORCH_NPU_BUG`
  证据更偏向 `ascend-pytorch/`、`torch_npu` glue 层或相关适配路径问题。
- `OPERATOR_BUG`
  证据更偏向底层算子、kernel、ACL/ACLNN、op-plugin 或更靠近算子实现层的问题。
- `API_BEHAVIOR_MISMATCH`
  API 行为和当前测试/预期不一致，但暂时不能稳定证明是测试问题还是源码问题。
- `FLAKY_OR_UNSTABLE`
  当前失败表现出明显不稳定、偶发或构建波动。
- `INSUFFICIENT_COVERAGE`
  测试可运行，但覆盖维度或异常/边界场景说明明显不足。
- `UNKNOWN`
  当前证据不足，无法可靠分类。

## Default Fix Mapping

默认修复映射如下：

- `TEST_BUG` -> `adjust_test`
- `ENVIRONMENT_MISSING` -> `manual_followup`
- `UNSUPPORTED_ON_NPU` -> `manual_followup`
- `OPERATOR_BUG` -> `manual_followup`
- `FLAKY_OR_UNSTABLE` -> `manual_followup`
- `INSUFFICIENT_COVERAGE` -> `manual_followup`
- `PYTORCH_BUG` -> `patch_pytorch`，仅 `--fix-mode safe`
- `TORCH_NPU_BUG` -> `patch_torch_npu`，仅 `--fix-mode safe`
- `API_BEHAVIOR_MISMATCH` -> `manual_followup`
- `UNKNOWN` -> `manual_followup`

核心原则：

- `tests` 模式下只自动修 `TEST_BUG`
- `safe` 模式只在证据比较明确时，允许低风险源码修复
- 环境问题、后端不支持、底层算子问题默认先报告，不强行改测试伪造通过

## Practical Notes

- 分类是 triage，不是根因证明。
- `analysis-engine=codex` 时，Codex 会结合日志和测试文件输出 triage。
- `analysis-engine=heuristic` 时，只使用本地启发式规则分类。
- 即使最终修复后通过，`results.*` 里也会保留初始失败分类字段，方便回看这次失败最初被判成什么。
