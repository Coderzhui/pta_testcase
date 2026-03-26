“””
测试目的：验证 `torch.utils._python_dispatch.is_traceable_wrapper_subclass` 在 NPU 环境下的完整功能行为。
API 名称：torch.utils._python_dispatch.is_traceable_wrapper_subclass

覆盖的入参维度：
- 参数传参与不传参：该 API 仅接收 1 个位置参数，无可选入参。
- 参数为 None / 非 None：已覆盖 None、普通 Tensor、Tensor 子类与普通对象。
- 枚举选项：不适用，该 API 无枚举参数。
- 多类型输入：已覆盖 torch.Tensor、Tensor 子类、普通对象。
- 正常输入：已覆盖可追踪 wrapper 子类实例。
- 异常输入：该 API 对非法输入不抛异常，仅返回布尔值。
- 边界值和等价类：已覆盖具有必要魔术方法的 wrapper 子类和缺少魔术方法的普通子类。

未覆盖项及原因：
- 无，已满足 API 功能一致性标准。
“””

import pytest

import torch
import torch_npu  # noqa: F401

from torch.utils import _python_dispatch as pd


def _require_npu():
    """检查 NPU 是否可用，不可用则跳过测试。"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为")


class PlainSubclass(torch.Tensor):
    """普通 Tensor 子类，不实现 wrapper 所需的魔术方法。"""
    pass


class Traceable(torch.Tensor):
    """
    可追踪的 wrapper 子类，实现了所有必要的魔术方法。
    包括：__tensor_flatten__、__tensor_unflatten__、__torch_dispatch__
    """
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )

    def __init__(self, elem):
        self.elem = elem

    def __tensor_flatten__(self):
        return ["elem"], {"device": str(self.elem.device)}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return Traceable(inner_tensors["elem"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def test_is_traceable_wrapper_subclass_true_for_traceable_npu_tensor():
    """
    测试用例：验证可追踪 wrapper 子类返回 True
    覆盖场景：
    - 在 NPU 上创建 Traceable wrapper
    - 验证返回值为 True
    - 验证返回值类型为 bool
    - 验证 wrapper 设备为 NPU
    """
    _require_npu()

    wrapped = Traceable(torch.ones(2, device="npu"))

    result = pd.is_traceable_wrapper_subclass(wrapped)

    assert result is True
    assert isinstance(result, bool)
    assert wrapped.device.type == "npu"


def test_is_traceable_wrapper_subclass_false_for_plain_npu_tensor():
    """
    测试用例：验证普通 Tensor 返回 False
    覆盖场景：
    - 在 NPU 上创建普通 tensor
    - 验证返回值为 False
    - 验证返回值类型为 bool
    """
    _require_npu()

    tensor = torch.ones(2, device="npu")

    result = pd.is_traceable_wrapper_subclass(tensor)

    assert result is False
    assert isinstance(result, bool)
    assert tensor.device.type == "npu"


def test_is_traceable_wrapper_subclass_false_for_plain_tensor_subclass_type():
    """
    测试用例：验证普通 Tensor 子类类型返回 False
    覆盖场景：
    - 传入类型对象（而非实例）
    - 验证返回值为 False
    """
    _require_npu()

    result = pd.is_traceable_wrapper_subclass(PlainSubclass)

    assert result is False
    assert isinstance(result, bool)


def test_is_traceable_wrapper_subclass_false_for_plain_tensor_subclass_instance():
    """
    测试用例：验证普通 Tensor 子类实例返回 False
    覆盖场景：
    - 在 NPU 上创建普通子类实例
    - 验证返回值为 False
    - 验证子类设备为 NPU
    """
    _require_npu()

    subclass = PlainSubclass(torch.ones(2, device="npu"))

    result = pd.is_traceable_wrapper_subclass(subclass)

    assert result is False
    assert isinstance(result, bool)
    assert subclass.device.type == "npu"


@pytest.mark.parametrize("value", [None, object(), 123, "tensor"])
def test_is_traceable_wrapper_subclass_false_for_non_tensor_objects(value):
    """
    测试用例：验证非 Tensor 对象返回 False
    覆盖场景：
    - 参数化测试多种非 Tensor 类型
    - None
    - 普通对象
    - 整数
    - 字符串
    - 验证所有情况返回 False
    """
    _require_npu()

    result = pd.is_traceable_wrapper_subclass(value)

    assert result is False
    assert isinstance(result, bool)
