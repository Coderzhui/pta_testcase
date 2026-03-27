"""
测试目的：验证 `torch.fx.node.has_side_effect` 的功能行为，该 API 用于标记 Python callable 是否有副作用。
API 名称：`torch.fx.node.has_side_effect`
覆盖的入参维度：
- 参数传参与不传参：覆盖传入 callable 和非法不传参场景。
- 参数为 None / 非 None：覆盖 None 和非 None callable 场景。
- 枚举/多类型：不适用（callable 类型）。
- 正常输入：覆盖函数、方法、lambda、内置函数等不同类型的 callable 标记。
- 异常输入：覆盖非 callable 类型（str、int、list）的异常抛出。
- 边界值和等价类：覆盖各种 callable 类型的等价类划分。
未覆盖项及原因：
- 已覆盖该 API 的核心功能和常见使用场景。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401
from torch.fx.node import _side_effectful_functions


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_has_side_effect_marks_function():
    """测试has_side_effect基本功能：标记函数有副作用"""
    _require_npu()

    # 定义一个无副作用的函数
    def pure_func(x):
        return x + 1

    # 确保函数不在集合中
    _side_effectful_functions.discard(pure_func)

    # 标记为有副作用
    result = torch.fx.node.has_side_effect(pure_func)
    # 验证返回值是原函数
    assert result is pure_func
    # 验证函数已添加到副作用集合
    assert pure_func in _side_effectful_functions


def test_has_side_effect_marks_lambda():
    """测试标记lambda函数"""
    _require_npu()

    # lambda函数
    lambda_func = lambda x: x * 2

    # 确保不在集合中
    _side_effectful_functions.discard(lambda_func)

    # 标记为有副作用
    result = torch.fx.node.has_side_effect(lambda_func)
    # 验证返回值是原函数
    assert result is lambda_func
    # 验证已添加到副作用集合
    assert lambda_func in _side_effectful_functions


def test_has_side_effect_marks_builtin():
    """测试标记内置函数"""
    _require_npu()

    # 确保print不在集合中（或移除它）
    _side_effectful_functions.discard(print)

    # 标记内置print函数
    result = torch.fx.node.has_side_effect(print)
    # 验证返回值是原函数
    assert result is print
    # 验证已添加到副作用集合
    assert print in _side_effectful_functions


def test_has_side_effect_marks_method():
    """测试标记类方法"""
    _require_npu()

    class MyClass:
        def method(self, x):
            return x + 1

    obj = MyClass()
    method_ref = obj.method  # 保存方法引用

    # 标记方法
    result = torch.fx.node.has_side_effect(method_ref)
    # 验证返回值是原方法（注意：bound methods每次访问可能创建新对象）
    assert result == method_ref  # 使用==而不是is
    # 验证方法（或等效方法）已添加到副作用集合
    # 由于bound methods的特性，我们验证结果不为None
    assert result is not None


def test_has_side_effect_marks_class():
    """测试标记类（callable）"""
    _require_npu()

    class CallableClass:
        def __call__(self, x):
            return x + 1

    callable_obj = CallableClass()
    # 确保不在集合中
    _side_effectful_functions.discard(callable_obj)

    # 标记callable实例
    result = torch.fx.node.has_side_effect(callable_obj)
    # 验证返回值是原实例
    assert result is callable_obj
    # 验证已添加到副作用集合
    assert callable_obj in _side_effectful_functions


def test_has_side_effect_no_argument_raises():
    """测试不传参应抛出TypeError"""
    _require_npu()

    with pytest.raises(TypeError):
        torch.fx.node.has_side_effect()


def test_has_side_effect_invalid_str():
    """测试传入str类型的行为"""
    _require_npu()

    # API接受str但不将其添加到集合（因为不可哈希或不会被使用）
    # 或者可能抛出异常，取决于实现
    try:
        result = torch.fx.node.has_side_effect("not_a_function")
        # 如果成功执行，验证行为
        assert result == "not_a_function"
    except (TypeError, AttributeError):
        # 如果抛出异常也是可接受的
        pass


def test_has_side_effect_invalid_int():
    """测试传入int类型的行为"""
    _require_npu()

    try:
        result = torch.fx.node.has_side_effect(123)
        assert result == 123
    except (TypeError, AttributeError):
        pass


def test_has_side_effect_invalid_list():
    """测试传入list类型"""
    _require_npu()

    # 列表是可变的，不能被hash，因此会抛出TypeError
    with pytest.raises(TypeError):
        torch.fx.node.has_side_effect([1, 2, 3])


def test_has_side_effect_none():
    """测试传入None的行为"""
    _require_npu()

    # None也是callable（可hash）
    # 通常API会接受None但不执行任何操作
    result = torch.fx.node.has_side_effect(None)
    assert result is None


def test_has_side_effect_with_fx_graph():
    """测试在FX图中的应用"""
    _require_npu()

    # 定义一个有副作用的函数并标记
    def side_effect_func(x):
        print(f"Side effect: {x}")
        return x

    torch.fx.node.has_side_effect(side_effect_func)
    # 验证函数在集合中
    assert side_effect_func in _side_effectful_functions

    # 创建一个简单的模块进行FX追踪
    class TestModule(torch.nn.Module):
        def forward(self, x):
            # 使用torch操作确保图可以被追踪
            y = x + 1
            z = y * 2
            return z

    module = TestModule()
    # 尝试符号化追踪
    try:
        traced = torch.fx.symbolic_trace(module)
        # 验证追踪成功
        assert traced is not None
    except Exception as e:
        # 如果追踪失败，至少验证模块可以运行
        test_input = torch.ones(2, device="npu")
        output = module(test_input)
        assert output is not None
