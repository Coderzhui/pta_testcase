#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:?usage: $0 <api_manifest.csv>}"
BASENAME="$(basename "$CSV_PATH" .csv)"
mkdir -p logs

cat <<EOF | codex exec \
  --cd . \
  --dangerously-bypass-approvals-and-sandbox \
  --output-last-message "logs/${BASENAME}_summary.md" \
  -
使用 batch-npu-api-test skill。

处理 CSV 文件：${CSV_PATH}

执行完整批处理，不要把任务拆成需要我再次确认的多轮对话。
要求：
1. 读取 CSV 中 pending 的 API
2. 启动 api_test_generator 子代理并行生成测试文件
3. 等待全部完成
4. 启动 api_test_reviewer 子代理并行审查
5. 对不通过项做最小修复
6. 仅对本次触达的文件运行 pytest
7. 输出最终总结到最终回复中

限制：
- 只修改 test/api_test/ 下的测试文件
- 不重写已通过文件
- 不做额外重构
- 若某 API 无法可靠执行，使用 skip/xfail 并说明原因
EOF