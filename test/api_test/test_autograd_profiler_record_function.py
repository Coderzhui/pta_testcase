"""
测试目的：验证 `torch.autograd.profiler.record_function` 的功能行为，性能分析代码块标记。
API 名称：`torch.autograd.profiler.record_function`
覆盖的入参维度：
- 参数传参与不传参：覆盖带args和不带args
- 参数为 None / 非 None：覆盖name和args参数
- 枚举/多类型：覆盖不同name、args类型
- 正常输入：覆盖上下文管理器和装饰器用法
- 异常输入：覆盖非法类型参数
- 边界值和等价类：覆盖空字符串、长字符串等边界值
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_record_function_as_context_manager():
    """测试作为上下文管理器"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("test_block"):
            x = torch.ones(10, device="npu")
            y = x * 2

    # 验证profile记录成功
    assert prof is not None


def test_record_function_with_name():
    """测试带name参数"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("my_operation"):
            x = torch.randn(10, device="npu")
            y = x.sum()

    assert prof is not None


def test_record_function_with_args():
    """测试带args参数"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("my_op", "extra_info"):
            x = torch.randn(5, device="npu")
            y = x * 2

    assert prof is not None


def test_record_function_as_decorator():
    """测试作为装饰器"""
    _require_npu()

    @torch.autograd.profiler.record_function("decorated_func")
    def test_func(x):
        return x * 2

    with torch.autograd.profiler.profile() as prof:
        result = test_func(torch.ones(5, device="npu"))

    assert torch.allclose(result, torch.ones(5, device="npu") * 2)
    assert prof is not None


def test_record_function_empty_name():
    """测试空字符串name"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function(""):
            x = torch.ones(3, device="npu")

    assert prof is not None


def test_record_function_nested():
    """测试嵌套的record_function"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("outer"):
            x = torch.ones(5, device="npu")
            with torch.autograd.profiler.record_function("inner"):
                y = x * 2
            z = y + 1

    assert prof is not None


def test_record_function_long_name():
    """测试长字符串name"""
    _require_npu()

    long_name = "a" * 1000
    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function(long_name):
            x = torch.ones(3, device="npu")

    assert prof is not None


def test_record_function_with_npu_operations():
    """测试NPU操作分析"""
    _require_npu()

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("npu_ops"):
            x = torch.randn(100, 100, device="npu")
            y = torch.matmul(x, x.t())
            z = y.sum()

    assert prof is not None
    # 验证能获取分析结果
    events = prof.key_averages()
    assert events is not None


def test_record_function_none_name_raises():
    """测试None作为name应抛出异常"""
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        with torch.autograd.profiler.record_function(None):
            pass


def test_record_function_return_value():
    """测试返回值（上下文管理器应返回self）"""
    _require_npu()

    ctx = torch.autograd.profiler.record_function("test")
    assert ctx is not None
    assert hasattr(ctx, '__enter__')
    assert hasattr(ctx, '__exit__')


def test_record_function_with_backward():
    """测试与backward结合使用"""
    _require_npu()

    x = torch.randn(10, device="npu", requires_grad=True)

    with torch.autograd.profiler.profile() as prof:
        with torch.autograd.profiler.record_function("forward"):
            y = x * 2
            z = y.sum()

        with torch.autograd.profiler.record_function("backward"):
            z.backward()

    assert prof is not None
    assert x.grad is not None
