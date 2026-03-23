---
name: batch-npu-api-test
description: 
  批量处理 PyTorch API 的 NPU 功能测试生成任务。当输入是 api_manifest.csv
  或者用户要求批量生成/审查/修复 test/api_test 下的 API 测试文件时使用。
---

你负责执行完整的批处理流水线，不要要求用户分步下达指令。

输入：
- 一个 CSV 文件路径，CSV 至少包含：
  - canonical_name
  - file_name
  - status
  - notes

工作流：
1. 读取 CSV 中 status=pending 的行。
2. 对每一行启动一个 api_test_generator 子代理并行生成测试文件。
3. 等待全部 generator 完成，收集生成结果。
4. 对新生成或修改过的测试文件启动 api_test_reviewer 子代理并行审查。
5. 对 reviewer 判定为不通过的文件，进行最小修复：
   - 启动 api_test_fixer 子代理处理单文件修复
   - 只修复失败项
   - 不重写已通过文件
   - 不改动 test/api_test 之外的业务代码
6. 对本次触达的测试文件运行 pytest。
7. 输出最终汇总：
   - 成功生成并通过的 API
   - 经修复后通过的 API
   - 仍失败 / skip 的 API 及原因
   - 明显未覆盖项

约束：
- 所有测试文件必须位于 test/api_test/
- 文件名必须使用 CSV 中的 file_name
- 测试必须运行在 NPU 上，使用 torch_npu
- 关注功能行为和接口覆盖，不做数值精度校验
- 异常场景必须使用 pytest.raises
- 禁止使用 pytest.xfail
- 只有环境缺失或当前 NPU 后端明确不支持时才允许 pytest.skip
- 文件头注释必须写明测试目的、API 名称、覆盖入参
