已按 `single-api-fix` 执行最小修复，只修改了允许范围内的 [test/api_test/test_nn_Module_named_parameters.py:49](/home/l00913161/projects/pta_testcase/test/api_test/test_nn_Module_named_parameters.py#L49)。变更是将无效 `self` 场景的断言从 `TypeError` 调整为实际稳定抛出的 `AttributeError`，并用 `match="_named_members"` 限定异常来源。

未运行 `pytest`，符合你的要求。剩余风险是该断言依赖当前 PyTorch 的内部报错路径；如果上游未来改成更早做类型校验，异常类型或消息可能再次变化。