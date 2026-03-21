# Failure Taxonomy

流水线会把每个 API 的最终状态映射到统一的 failure category，供 `results.json`、`results.csv` 和 `summary.md` 使用。

## Categories

- `NONE`: 测试最终通过，没有需要继续归因的问题。
- `TEST_BUG`: 生成出的测试文件本身有问题，通常应在 `test/api_test/` 内修复。
- `UNSUPPORTED_ON_NPU`: 当前 NPU 后端、当前构建或当前 dispatch/layout 组合不支持该路径。
- `ENVIRONMENT_MISSING`: 环境缺少 `torch_npu`、NPU 不可用或基础运行条件不满足。
- `API_BEHAVIOR_MISMATCH`: API 行为与测试预期不一致，但还不能稳定归类为测试问题或源码问题。
- `PYTORCH_BUG`: 证据更偏向 `pytorch/` 内部实现问题。
- `TORCH_NPU_BUG`: 证据更偏向 `ascend-pytorch/` 或 `torch_npu` 路径问题。
- `FLAKY_OR_UNSTABLE`: 当前能力存在不稳定、偶发或构建相关波动。
- `INSUFFICIENT_COVERAGE`: 测试可以运行，但覆盖说明或入参维度覆盖明显不足。
- `UNKNOWN`: 还无法自动可靠分类。

## Default Fix Mapping

- `TEST_BUG` -> `adjust_test`
- `ENVIRONMENT_MISSING` -> `adjust_test`，优先补 `pytest.skip`
- `UNSUPPORTED_ON_NPU` -> `adjust_test`，优先补 `pytest.skip/xfail`
- `FLAKY_OR_UNSTABLE` -> `adjust_test`，优先补 `pytest.xfail`
- `PYTORCH_BUG` -> `patch_pytorch`，仅 `--fix-mode safe`
- `TORCH_NPU_BUG` -> `patch_torch_npu`，仅 `--fix-mode safe`
- `API_BEHAVIOR_MISMATCH` -> `adjust_test_or_patch_source`，仅 `--fix-mode safe`

## Notes

- 分类是启发式的，不是根因证明。
- 自动修复只针对低风险、局部、可复跑验证的问题。
- 一旦触发源码修复，报告里必须留下 patch 摘要和二次回归结果。
