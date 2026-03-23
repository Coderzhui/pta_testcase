---
name: single-api-fix
description: 
  处理单个 PyTorch API 的修复请求。根据 fix request JSON 选择 api_test_fixer
  或 api_safe_fixer，执行最小修复，不运行 pytest。
---

你负责执行单个 API 的修复，不要要求用户分步确认。

输入：
- 一个 fix request JSON 文件，至少包含：
  - canonical_name
  - file_name
  - fix_mode
  - failure_category
  - fix_recommendation
  - final_status
  - pytest_outcome
  - allowed_scopes
  - root_cause_summary
  - failure_messages

工作流：
1. 读取 fix request JSON。
2. 如果 `fix_mode` 为 `safe` 且 `allowed_scopes` 中包含 `pytorch/` 或 `ascend-pytorch/`，启动 `api_safe_fixer`。
3. 否则启动 `api_test_fixer`。
4. 只做最小修复，不运行 pytest。
5. 最终输出修复摘要、修改文件、剩余风险。

约束：
- 禁止使用 pytest.xfail
- 只有环境缺失或当前 NPU 后端明确不支持时才允许 pytest.skip
- 不得触达 `allowed_scopes` 之外的文件
- 不得伪造覆盖
