已按 `single-api-fix` 处理该单个请求，只修改了 [test_autograd_graph__MultiHandle.py](/home/l00913161/projects/pta_testcase/test/api_test/test_autograd_graph__MultiHandle.py)。

修复内容是把错误断言从“`_MultiHandle(None)` 构造时抛 `TypeError`”改为“对象可先构造，`remove()` 时抛 `TypeError`”，与当前实现一致；测试名同步调整为 `test_multihandle_none_handles_raise_on_remove`。未运行 `pytest`，回归留给外层 pipeline。