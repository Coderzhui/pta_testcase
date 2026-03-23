已按请求完成单 API 最小修复，只修改了 [test/api_test/test_nn_Module_register_load_state_dict_post_hook.py](/home/l00913161/projects/pta_testcase/test/api_test/test_nn_Module_register_load_state_dict_post_hook.py#L55)。

修复点是把异常用例改为符合当前 PyTorch 行为：`register_load_state_dict_post_hook(None)` 先注册成功，再在 `load_state_dict({}, strict=False)` 触发该 `None` hook 时断言 `TypeError`。未使用 `pytest.xfail`，未触达 `allowed_scopes` 之外的文件，也没有运行 `pytest`。

剩余风险：未做本地回归，最终结果依赖外层 pipeline 验证当前环境中的实际 NPU/PyTorch 行为。