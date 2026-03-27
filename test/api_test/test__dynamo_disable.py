"""
测试目的：验证 `torch._dynamo.disable` 的功能行为，禁用TorchDynamo的装饰器。
API 名称：`torch._dynamo.disable`
覆盖的入参维度：
- 参数传参与不传参：覆盖无参装饰器、带参数装饰器
- 参数为 None / 非 None：覆盖 fn=None 和 fn=callable
- 枚举/多类型：覆盖 recursive=True/False 选项
- 正常输入：覆盖函数装饰、递归/非递归模式
- 异常输入：覆盖非法类型参数
- 边界值和等价类：覆盖 recursive 参数的 True/False 等价类
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_dynamo_disable_no_args():
    """测试无参装饰器"""
    _require_npu()

    @torch._dynamo.disable
    def test_func(x):
        return x * 2

    # 验证函数可正常调用
    result = test_func(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") * 2)


def test_dynamo_disable_with_recursive_true():
    """测试 recursive=True 完全禁用"""
    _require_npu()

    @torch._dynamo.disable(recursive=True)
    def test_func(x):
        return x * 2

    result = test_func(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") * 2)


def test_dynamo_disable_with_recursive_false():
    """测试 recursive=False 部分禁用"""
    _require_npu()

    @torch._dynamo.disable(recursive=False)
    def test_func(x):
        return x * 2

    result = test_func(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") * 2)


def test_dynamo_disable_returns_callable():
    """测试装饰器返回可调用对象"""
    _require_npu()

    def original_func(x):
        return x + 1

    decorated = torch._dynamo.disable(original_func)
    assert callable(decorated)

    result = decorated(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") + 1)


def test_dynamo_disable_preserves_functionality():
    """测试装饰后函数功能保持正确"""
    _require_npu()

    @torch._dynamo.disable
    def complex_func(x):
        # 执行一些NPU操作
        y = x + 1
        z = y * 2
        return z - 1

    x = torch.ones(3, device="npu")
    result = complex_func(x)
    expected = (x + 1) * 2 - 1
    assert torch.allclose(result, expected)


def test_dynamo_disable_nested_function():
    """测试嵌套函数的装饰器"""
    _require_npu()

    @torch._dynamo.disable
    def outer_func(x):
        def inner_func(y):
            return y * 3
        return inner_func(x) + 1

    result = outer_func(torch.ones(2, device="npu"))
    expected = torch.ones(2, device="npu") * 3 + 1
    assert torch.allclose(result, expected)


def test_dynamo_disable_with_lambda():
    """测试装饰lambda函数"""
    _require_npu()

    # 装饰lambda
    disabled_lambda = torch._dynamo.disable(lambda x: x * 2)
    result = disabled_lambda(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") * 2)


def test_dynamo_disable_invalid_non_callable_raises():
    """测试装饰非callable应抛出异常"""
    _require_npu()

    # API使用assert callable(fn)，因此抛出AssertionError
    with pytest.raises(AssertionError):
        torch._dynamo.disable("not_a_function")


def test_dynamo_disable_on_method():
    """测试在类方法上使用装饰器"""
    _require_npu()

    class TestClass:
        @torch._dynamo.disable
        def method(self, x):
            return x * 2

    obj = TestClass()
    result = obj.method(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") * 2)


def test_dynamo_disable_with_none_fn():
    """测试 fn=None 时返回部分应用的装饰器"""
    _require_npu()

    # 当 fn=None 时，应返回一个部分应用的装饰器函数
    decorator = torch._dynamo.disable(fn=None, recursive=True)
    assert callable(decorator)

    # 使用返回的装饰器
    @decorator
    def test_func(x):
        return x + 1

    result = test_func(torch.ones(2, device="npu"))
    assert torch.allclose(result, torch.ones(2, device="npu") + 1)
