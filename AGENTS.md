# AGENTS.md

## Repository goal
批量为 PyTorch API 生成 NPU 功能测试用例，并自动完成运行、结果分析、报告输出，以及低风险自动修复。

## Hard requirements
- 每个 API 只生成 1 个测试文件
- 测试文件统一放在 `test/api_test/`
- 文件名必须严格使用 CSV 中提供的 `file_name`
- 测试框架必须是 `pytest`
- 测试必须运行在 NPU 上，使用 `torch_npu`
- 测试关注 API 功能行为与接口覆盖，不做数值精度比对
- 文件头部注释必须说明：
  - 测试目的
  - API 名称
  - 覆盖的入参维度

## Coverage rules
必须尽量覆盖该 API 的所有入参维度（按实际签名裁剪）：
1. 参数传参与不传参
2. 参数为 None / 非 None
3. 枚举选项的所有主要候选值
4. 支持多类型时覆盖主要类型
5. 正常输入
6. 异常输入
7. 边界值和等价类

## Assertion rules
断言聚焦于：
- API 可调用
- 返回对象存在，类型合理
- 输出设备行为符合预期（NPU）
- 异常场景使用 `pytest.raises`
- 不要求校验具体数值正确性

## Failure handling
- 如果某个 API 在当前环境无法可靠构造最小可运行用例，使用 `pytest.skip` 或 `pytest.xfail`
- 必须写清楚原因
- 不要伪造覆盖

## Pipeline stages
- manifest / report / orchestration 阶段允许修改仓库中的脚本和文档，用于支撑批处理流水线
- generator / reviewer 阶段默认只修改 `test/api_test/` 下 CSV 对应的目标文件
- safe fix 阶段仅在明确低风险且可直接回归验证时，允许修改 `pytorch/` 或 `ascend-pytorch/`

## File discipline
- 生成测试时，每个 API 仍然只允许对应 1 个 `test/api_test/` 下的目标文件
- 不要顺手修改与当前失败无关的代码
- 自动修复源码问题时必须保持最小改动，并在报告里留下修复摘要和 rerun 结果
- 生成后优先保证文件可 import、pytest 可收集、命名正确
