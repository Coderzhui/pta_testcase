# 测试目的：
# 1. 验证 `Tensor.new_empty` 在 NPU 上可调用，且返回 Tensor 的设备仍为 NPU。
# 2. 覆盖参数省略、显式 None、显式非 None、主要 dtype、主要布局、边界形状和异常输入。
# API 名称：`Tensor.new_empty`
#
# 覆盖的入参维度：
# | 维度 | 覆盖方式 |
# | --- | --- |
# | 参数传参与不传参 | 省略 `dtype/device/requires_grad/layout`，以及显式传入这些参数 |
# | 参数为 None / 非 None | `dtype=None`、`device=None`，以及 `dtype=torch.float16`、`device=torch.device("npu")` |
# | 枚举选项主要候选值 | `layout=torch.strided` |
# | 支持多类型时覆盖主要类型 | `size` 使用 `tuple` / `list` |
# | 正常输入 | 常规形状 `(3, 4)`、`(1,)` |
# | 异常输入 | `size="bad"` 触发类型异常 |
# | 边界值和等价类 | 空维度 `0`、单元素维度 `1` |
#
# 未覆盖项及原因：
# | 未覆盖项 | 原因 |
# | --- | --- |
# | `pin_memory=True` | NPU 场景下该参数依赖主机 pinned memory 能力，且与本 API 的核心 NPU 功能无关，缺少稳定最小用例 |
# | 非 `torch.strided` 布局 | 当前 API 在 NPU 上仅对 `strided` 路径有稳定最小用例，其他布局缺少可靠构造方式 |

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu() -> None:
    if not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用的 NPU，无法验证 Tensor.new_empty 的 NPU 行为。")


@pytest.mark.parametrize(
    "size, kwargs, expected_dtype, expected_requires_grad",
    [
        ((3, 4), {}, torch.float32, False),
        ((1,), {"dtype": None, "device": None}, torch.float32, False),
        (
            [2, 0],
            {
                "dtype": torch.float16,
                "device": torch.device("npu"),
                "requires_grad": True,
                "layout": torch.strided,
            },
            torch.float16,
            True,
        ),
    ],
)
def test_tensor_new_empty_on_npu(size, kwargs, expected_dtype, expected_requires_grad):
    _require_npu()

    base = torch.randn(2, device="npu")
    result = base.new_empty(size, **kwargs)

    assert isinstance(result, torch.Tensor)
    assert result.device.type == "npu"
    assert str(result.device).startswith("npu")
    assert result.dtype == expected_dtype
    assert result.requires_grad == expected_requires_grad
    assert result.layout == torch.strided
    assert tuple(result.shape) == tuple(size)


def test_tensor_new_empty_invalid_size_raises():
    _require_npu()

    base = torch.randn(2, device="npu")

    with pytest.raises(TypeError):
        base.new_empty("bad")
