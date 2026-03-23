已按 `single-api-fix` 流程完成修复，只触达允许范围内的 [test/api_test/test_Tensor_register_hook.py](test/api_test/test_Tensor_register_hook.py)。

修复摘要：将 `register_hook(None)` 的异常断言从“注册时抛 `TypeError`”改为“在 `backward()` 触发 hook 调用时抛 `TypeError`”，与请求文件给出的实际 PyTorch 行为一致；未使用 `pytest.xfail`，也没有运行 `pytest`。

剩余风险：结果尚未回归验证，留给外层 pipeline 执行。