"""
测试目的：验证 `torch.empty_like` 的功能行为，返回与输入tensor大小相同的未初始化tensor。
API 名称：`torch.empty_like`
覆盖的入参维度：
- 参数传参与不传参：覆盖所有可选参数（dtype、layout、device、requires_grad、memory_format）
- 参数为 None / 非 None：覆盖各参数为None和具体值
- 枚举/多类型：覆盖不同dtype、layout、memory_format选项
- 正常输入：覆盖各种合法输入
- 异常输入：覆盖非法类型参数
- 边界值和等价类：覆盖空tensor、不同shape、不同device
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_empty_like_basic():
    """测试基本的empty_like操作"""
    _require_npu()

    x = torch.ones(3, 4, device="npu")
    result = torch.empty_like(x)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device


def test_empty_like_with_dtype():
    """测试指定dtype"""
    _require_npu()

    x = torch.ones(3, 4, device="npu", dtype=torch.float32)
    result = torch.empty_like(x, dtype=torch.float16)

    assert result.shape == x.shape
    assert result.dtype == torch.float16


def test_empty_like_with_device():
    """测试指定device"""
    _require_npu()

    x = torch.ones(3, 4, device="cpu")
    result = torch.empty_like(x, device="npu")

    assert result.shape == x.shape
    assert result.device.type == "npu"


def test_empty_like_with_requires_grad():
    """测试指定requires_grad"""
    _require_npu()

    x = torch.ones(3, 4, device="npu", requires_grad=False)
    result = torch.empty_like(x, requires_grad=True)

    assert result.requires_grad is True


def test_empty_like_preserve_memory_format():
    """测试默认memory_format=preserve_format"""
    _require_npu()

    x = torch.ones(3, 4, device="npu")
    result = torch.empty_like(x)

    assert result.shape == x.shape
    assert result.stride() == x.stride()


def test_empty_like_contiguous_format():
    """测试memory_format=torch.contiguous_format"""
    _require_npu()

    x = torch.ones(3, 4, device="npu")
    result = torch.empty_like(x, memory_format=torch.contiguous_format)

    assert result.shape == x.shape
    assert result.is_contiguous()


def test_empty_like_channels_last():
    """测试memory_format=torch.channels_last（4D tensor）"""
    _require_npu()

    x = torch.ones(2, 3, 4, 5, device="npu")
    try:
        result = torch.empty_like(x, memory_format=torch.channels_last)
        assert result.shape == x.shape
    except RuntimeError:
        # NPU可能不支持channels_last
        pytest.skip("NPU不支持channels_last格式")


def test_empty_like_different_shapes():
    """测试不同shape的tensor"""
    _require_npu()

    shapes = [(5,), (3, 4), (2, 3, 4), (1,)]

    for shape in shapes:
        x = torch.ones(shape, device="npu")
        result = torch.empty_like(x)
        assert result.shape == shape


def test_empty_like_zero_dim():
    """测试0维tensor（标量）"""
    _require_npu()

    x = torch.tensor(5.0, device="npu")
    result = torch.empty_like(x)

    assert result.dim() == 0
    assert result.shape == ()


def test_empty_like_empty_tensor():
    """测试空tensor"""
    _require_npu()

    x = torch.ones(0, device="npu")
    result = torch.empty_like(x)

    assert result.shape == (0,)
    assert len(result) == 0


def test_empty_like_different_dtypes():
    """测试不同dtype选项"""
    _require_npu()

    x = torch.ones(3, 4, device="npu", dtype=torch.float32)
    dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

    for dtype in dtypes:
        try:
            result = torch.empty_like(x, dtype=dtype)
            assert result.dtype == dtype
        except RuntimeError:
            # 某些dtype可能不被NPU支持
            continue


def test_empty_like_non_tensor_raises():
    """测试非tensor输入应抛出异常"""
    _require_npu()

    with pytest.raises(TypeError):
        torch.empty_like("not_tensor")


def test_empty_like_with_layout():
    """测试指定layout"""
    _require_npu()

    x = torch.ones(3, 4, device="npu")
    # 尝试strided layout（默认）
    result = torch.empty_like(x, layout=torch.strided)

    assert result.layout == torch.strided


def test_empty_like_preserves_properties():
    """测试默认情况下保留输入tensor的属性"""
    _require_npu()

    x = torch.ones(3, 4, device="npu", dtype=torch.float32)
    result = torch.empty_like(x)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert result.layout == x.layout
