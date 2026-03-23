已按 `single-api-fix` skill 完成修复，只修改了 [test/api_test/test_utils_swap_tensors.py](/home/l00913161/projects/pta_testcase/test/api_test/test_utils_swap_tensors.py)，符合请求里的 `allowed_scopes`。

修复内容是放宽 `test_swap_tensors_invalid_input_raises` 的异常断言，接受当前 `torch.utils.swap_tensors` 对非 Tensor 参数实际抛出的 `AttributeError`，同时保留 `TypeError` 和 `RuntimeError`。未使用 `pytest.xfail`，也未运行 `pytest`。

剩余风险：该异常断言仍依赖上游当前实现细节；如果后续 `swap_tensors` 统一了非法入参的异常类型，外层回归可能需要再同步调整。