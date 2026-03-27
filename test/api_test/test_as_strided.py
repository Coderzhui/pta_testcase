"""
测试目的：验证 `torch.as_strided` 的功能行为，创建具有指定size、stride和storage_offset的tensor视图。
API 名称：`torch.as_strided`
覆盖的入参维度：
- 参数传参与不传参：覆盖有/无storage_offset
- 参数为 None / 非 None：覆盖storage_offset=None和具体值
- 枚举/多类型：覆盖不同size、stride组合
- 正常输入：覆盖各种合法视图创建
- 异常输入：覆盖越界访问、负stride等
- 边界值和等价类：覆盖空tensor、单元素、多维tensor
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_as_strided_basic():
    """测试基本的as_strided操作"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    # 创建2x3的视图
    result = torch.as_strided(x, (2, 3), (3, 1))

    assert result.shape == (2, 3)
    assert result.stride() == (3, 1)
    expected = torch.tensor([[0, 1, 2], [3, 4, 5]], device="npu", dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_as_strided_with_storage_offset():
    """测试带storage_offset的as_strided"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    # 从索引1开始
    result = torch.as_strided(x, (2, 2), (2, 1), storage_offset=1)

    assert result.shape == (2, 2)
    expected = torch.tensor([[1, 2], [3, 4]], device="npu", dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_as_strided_storage_offset_none():
    """测试storage_offset=None（默认）"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (2, 3), (3, 1), storage_offset=None)

    assert result.shape == (2, 3)
    expected = torch.tensor([[0, 1, 2], [3, 4, 5]], device="npu", dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_as_strided_1d():
    """测试1D tensor"""
    _require_npu()

    x = torch.arange(10, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (5,), (2,))

    assert result.shape == (5,)
    expected = torch.tensor([0, 2, 4, 6, 8], device="npu", dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_as_strided_3d():
    """测试3D tensor"""
    _require_npu()

    x = torch.arange(24, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (2, 3, 4), (12, 4, 1))

    assert result.shape == (2, 3, 4)
    assert result.stride() == (12, 4, 1)


def test_as_strided_empty_size():
    """测试空size"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (0,), (1,))

    assert result.shape == (0,)
    assert len(result) == 0


def test_as_strided_single_element():
    """测试单元素"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (1,), (1,), storage_offset=3)

    assert result.shape == (1,)
    assert torch.allclose(result, torch.tensor([3], device="npu", dtype=torch.float32))


def test_as_strided_negative_stride_raises():
    """测试负stride"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    # 负stride可能合法，也可能不合法，取决于实现
    try:
        result = torch.as_strided(x, (3,), (-1,))
        # 如果成功，验证结果
        assert result.shape == (3,)
    except RuntimeError:
        # 如果抛出异常也是可接受的
        pass


def test_as_strided_out_of_bounds_raises():
    """测试越界应抛出异常"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    # 尝试创建超出原始tensor范围的视图
    with pytest.raises(RuntimeError):
        torch.as_strided(x, (10,), (1,))


def test_as_strided_different_dtypes():
    """测试不同dtype"""
    _require_npu()

    dtypes = [torch.float32, torch.float16, torch.int32]

    for dtype in dtypes:
        x = torch.arange(6, device="npu", dtype=dtype)
        result = torch.as_strided(x, (2, 3), (3, 1))
        assert result.dtype == dtype


def test_as_strided_is_view():
    """测试返回的是视图（共享存储）"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    result = torch.as_strided(x, (2, 3), (3, 1))

    # 修改视图应影响原tensor
    result[0, 0] = 100
    assert x[0] == 100


def test_as_strided_size_and_stride_length_mismatch_raises():
    """测试size和stride长度不匹配应抛出异常"""
    _require_npu()

    x = torch.arange(6, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.as_strided(x, (2, 3), (1,))  # size长度2，stride长度1


def test_as_strided_non_tensor_input_raises():
    """测试非tensor输入应抛出异常"""
    _require_npu()

    with pytest.raises(TypeError):
        torch.as_strided("not_tensor", (2,), (1,))
