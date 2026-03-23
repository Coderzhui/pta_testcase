修复已完成，只改了允许范围内的单个文件：[test/api_test/test_nn_Module_modules.py:50](test/api_test/test_nn_Module_modules.py#L50)。

本次是最小修复：将 `test_module_modules_invalid_self_raises` 的异常断言从 `TypeError` 调整为实际稳定触发的 `AttributeError`，与请求中的根因描述一致。未使用 `pytest.xfail`，也未运行 `pytest`。

剩余风险：未做本地回归，最终通过情况依赖外层 pipeline 验证当前环境下的真实行为是否仍保持为 `AttributeError`。