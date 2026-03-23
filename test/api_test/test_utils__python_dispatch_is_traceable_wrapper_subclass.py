"""
测试目的：验证 `torch.utils._python_dispatch.is_traceable_wrapper_subclass` 在 NPU 环境下的
基础判定行为，覆盖可追踪 wrapper 子类、普通 Tensor、普通 Tensor 子类以及非 Tensor 对象。
API 名称：torch.utils._python_dispatch.is_traceable_wrapper_subclass

覆盖维度表：
| 维度 | 覆盖情况 | 说明 |
| --- | --- | --- |
| 参数传参与不传参 | 未覆盖 | 该 API 仅接收 1 个位置参数，无可选入参 |
| 参数为 None / 非 None | 部分覆盖 | 覆盖 `None`、普通 Tensor、Tensor 子类与普通对象 |
| 枚举选项 | 不适用 | 该 API 无枚举参数 |
| 多类型输入 | 已覆盖 | 覆盖 `torch.Tensor`、Tensor 子类、普通对象 |
| 正常输入 | 已覆盖 | 覆盖可追踪 wrapper 子类实例 |
| 异常输入 | 未覆盖 | 该 API 对非法输入不抛异常，仅返回布尔值 |
| 边界值和等价类 | 部分覆盖 | 覆盖“具有必要魔术方法的 wrapper 子类”和“缺少魔术方法的普通子类” |

未覆盖项及原因：
1. 参数传参与不传参：API 签名固定为单参数，没有可选参数可裁剪。
2. 枚举选项：API 不接受枚举入参。
3. 异常输入：该 API 对非法输入不抛异常，只返回 `False`，因此没有可用的 `pytest.raises` 分支。
4. 更复杂的 wrapper aliasing / tracing 场景：该 API 只做静态判定，当前用例已足够覆盖接口语义，无需构造更重的编译场景。
"""

import pytest

import torch
import torch_npu  # noqa: F401

from torch.utils import _python_dispatch as pd


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为")


class PlainSubclass(torch.Tensor):
    pass


class Traceable(torch.Tensor):
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
    _require_npu()

    wrapped = Traceable(torch.ones(2, device="npu"))

    result = pd.is_traceable_wrapper_subclass(wrapped)

    assert result is True
    assert isinstance(result, bool)
    assert wrapped.device.type == "npu"


def test_is_traceable_wrapper_subclass_false_for_plain_npu_tensor():
    _require_npu()

    tensor = torch.ones(2, device="npu")

    result = pd.is_traceable_wrapper_subclass(tensor)

    assert result is False
    assert isinstance(result, bool)
    assert tensor.device.type == "npu"


def test_is_traceable_wrapper_subclass_false_for_plain_tensor_subclass_type():
    _require_npu()

    result = pd.is_traceable_wrapper_subclass(PlainSubclass)

    assert result is False
    assert isinstance(result, bool)


def test_is_traceable_wrapper_subclass_false_for_plain_tensor_subclass_instance():
    _require_npu()

    subclass = PlainSubclass(torch.ones(2, device="npu"))

    result = pd.is_traceable_wrapper_subclass(subclass)

    assert result is False
    assert isinstance(result, bool)
    assert subclass.device.type == "npu"


@pytest.mark.parametrize("value", [None, object(), 123, "tensor"])
def test_is_traceable_wrapper_subclass_false_for_non_tensor_objects(value):
    _require_npu()

    result = pd.is_traceable_wrapper_subclass(value)

    assert result is False
    assert isinstance(result, bool)
