已按 `single-api-fix` 执行单 API 修复，只触达了允许范围内的 [test_nn_Module_buffers.py](test/api_test/test_nn_Module_buffers.py)。

修复内容是把 `test_module_buffers_invalid_self_raises` 对 `torch.nn.Module.buffers("bad")` 的异常预期从 `TypeError` 调整为 `AttributeError`，与请求文件里的稳定失败栈一致。未使用 `pytest.xfail`，也没有运行 `pytest`。

剩余风险：这个断言依赖当前上游实现对非法 `self` 通过访问 `self.named_buffers` 抛出 `AttributeError`；如果未来实现改成显式类型检查，异常类型可能再次变化。