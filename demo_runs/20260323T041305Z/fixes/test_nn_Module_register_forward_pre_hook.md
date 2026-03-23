已按 `single-api-fix` 流程完成单个 API 的最小修复，只修改了允许范围内的 [test/api_test/test_nn_Module_register_forward_pre_hook.py](/home/l00913161/projects/pta_testcase/test/api_test/test_nn_Module_register_forward_pre_hook.py)。

修复内容是把 `test_module_register_forward_pre_hook_invalid_hook_raises` 的断言从“`register_forward_pre_hook(None)` 注册时立即抛 `TypeError`”调整为“注册后在执行 `forward` 时触发 `TypeError`”，以匹配当前 PyTorch 的实际行为。未使用 `pytest.xfail`，也未运行 `pytest`。

剩余风险是外层回归仍需确认当前环境下该异常类型和触发时机没有框架分支差异。