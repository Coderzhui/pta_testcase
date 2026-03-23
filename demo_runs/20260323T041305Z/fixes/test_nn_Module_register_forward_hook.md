已按 `single-api-fix` 执行该单 API 修复，且只修改了允许范围内的文件：[test_nn_Module_register_forward_hook.py](/home/l00913161/projects/pta_testcase/test/api_test/test_nn_Module_register_forward_hook.py)。

修复内容是把 `register_forward_hook(None)` 的 `TypeError` 断言从“注册阶段”调整到“实际执行 `forward` 触发 hook 调用阶段”，与当前 PyTorch 的真实行为一致。未使用 `pytest.xfail`，也未运行 `pytest`。剩余风险仅是未做本地回归，需由外层 pipeline 验证该异常路径在当前环境中的实际表现。