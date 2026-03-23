"""
测试目的：验证 `Tensor.new_zeros` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`Tensor.new_zeros`
覆盖的入参维度：
- 参数传参与不传参：覆盖省略 `dtype/device/requires_grad/layout` 和显式传入这些参数。
- 参数为 None / 非 None：覆盖 `dtype=None`、`device=None` 和显式 `torch.float16`、`torch.device("npu")`。
- 枚举/多类型：覆盖 `layout=torch.strided`，以及 `size` 的 `tuple` / `list` 输入。
- 正常输入：覆盖常规形状、空维度和 requires_grad 切换。
- 异常输入：覆盖非法 `size` 类型。
- 边界值和等价类：覆盖 `0` 和 `1` 维。
未覆盖项及原因：
- `pin_memory=True` 依赖主机 pinned memory，不属于该 NPU 功能测试的稳定最小用例。
- 非 `strided` 布局在当前场景缺少稳定最小构造方式。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")

@pytest.mark.parametrize(
    "size, kwargs, expected_dtype, expected_requires_grad",
    [
        ((2, 3), {}, torch.float32, False),
        ((1,), {"dtype": None, "device": None}, torch.float32, False),
        ([2, 0], {"dtype": torch.float16, "device": torch.device("npu"), "requires_grad": True, "layout": torch.strided}, torch.float16, True),
    ],
)
def test_tensor_new_zeros_on_npu(size, kwargs, expected_dtype, expected_requires_grad):
    _require_npu()

    base = torch.randn(2, device="npu")
    result = base.new_zeros(size, **kwargs)

    assert isinstance(result, torch.Tensor)
    assert result.device.type == "npu"
    assert result.dtype == expected_dtype
    assert result.requires_grad == expected_requires_grad
    assert result.layout == torch.strided
    assert tuple(result.shape) == tuple(size)
    assert torch.count_nonzero(result.cpu()).item() == 0


def test_tensor_new_zeros_invalid_size_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch.ones(2, device="npu").new_zeros("bad")
