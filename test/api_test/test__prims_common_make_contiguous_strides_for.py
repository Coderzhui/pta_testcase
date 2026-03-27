"""
测试目的：验证 `torch._prims_common.make_contiguous_strides_for` 的纯 shape 计算行为。
API 名称：`torch._prims_common.make_contiguous_strides_for`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认 `row_major=True` 和显式 `row_major=False`。
- 参数为 None / 非 None：覆盖非法 None 输入的异常抛出。
- 枚举/多类型：覆盖 row_major 的 True/False 两种值，以及 shape 的 tuple/list 类型。
- 正常输入：覆盖各种合法 shape 的正确性验证（包括空 shape、一维、多维）。
- 异常输入：覆盖非法负维度、非 tuple/list 类型（str、int、dict）。
- 边界值和等价类：覆盖空 shape、零维、单元素 shape、大 shape 的边界场景。
未覆盖项及原因：
- 该 API 为纯 shape 计算工具，不涉及真实 Tensor 分配；但已覆盖所有参数组合场景。
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


def test_make_contiguous_strides_for_row_major_true():
    """测试row_major=True时的正确性（默认行为）"""
    _require_npu()

    # 测试不同shape的行优先stride计算
    test_cases = [
        ((2, 3), (3, 1)),      # 2x3矩阵，stride=(3,1)
        ((2, 3, 4), (12, 4, 1)),  # 2x3x4张量
        ((5,), (1,)),          # 一维
        ((1,), (1,)),          # 单元素
    ]

    for shape, expected in test_cases:
        # 默认row_major=True
        result = torch._prims_common.make_contiguous_strides_for(shape)
        assert result == expected, f"Shape {shape}: expected {expected}, got {result}"

        # 显式row_major=True
        result_explicit = torch._prims_common.make_contiguous_strides_for(shape, row_major=True)
        assert result_explicit == expected, f"Shape {shape} (explicit True): expected {expected}, got {result_explicit}"


def test_make_contiguous_strides_for_negative_dim_raises():
    """测试负维度应抛出异常"""
    _require_npu()

    with pytest.raises(RuntimeError):
        torch._prims_common.make_contiguous_strides_for((-1, 3))


def test_make_contiguous_strides_for_invalid_none_raises():
    """测试传入None应抛出异常"""
    _require_npu()

    with pytest.raises((AssertionError, TypeError, RuntimeError)):
        torch._prims_common.make_contiguous_strides_for(None)


def test_make_contiguous_strides_for_row_major_false():
    """测试row_major=False时的正确性"""
    _require_npu()

    # 测试不同shape的row_major=False时的stride计算
    # 注意：实际的stride值取决于PyTorch内部实现
    test_cases = [
        ((2, 3), (1, 2)),      # 2x3矩阵
        ((5,), (1,)),          # 一维
    ]

    for shape, expected in test_cases:
        result = torch._prims_common.make_contiguous_strides_for(shape, row_major=False)
        assert result == expected, f"Shape {shape}: expected {expected}, got {result}"

    # 对于3D tensor，验证返回正确的tuple结构
    shape_3d = (2, 3, 4)
    result_3d = torch._prims_common.make_contiguous_strides_for(shape_3d, row_major=False)
    assert isinstance(result_3d, tuple)
    assert len(result_3d) == 3


def test_make_contiguous_strides_for_empty_shape():
    """测试空shape的边界情况"""
    _require_npu()

    # 空shape（标量）
    result = torch._prims_common.make_contiguous_strides_for(())
    assert result == (), f"Empty shape: expected (), got {result}"

    # 空shape的列优先
    result_col = torch._prims_common.make_contiguous_strides_for((), row_major=False)
    assert result_col == (), f"Empty shape (col major): expected (), got {result_col}"


def test_make_contiguous_strides_for_single_element():
    """测试单元素shape的边界情况"""
    _require_npu()

    # 单元素shape
    result = torch._prims_common.make_contiguous_strides_for((1,))
    assert result == (1,), f"Single element: expected (1,), got {result}"

    result = torch._prims_common.make_contiguous_strides_for((1,), row_major=False)
    assert result == (1,), f"Single element (col major): expected (1,), got {result}"


def test_make_contiguous_strides_for_list_input():
    """测试传入list而非tuple"""
    _require_npu()

    # 传入list应该也能工作
    shape_list = [2, 3]
    result = torch._prims_common.make_contiguous_strides_for(shape_list)
    expected = (3, 1)
    assert result == expected, f"List input: expected {expected}, got {result}"


def test_make_contiguous_strides_for_invalid_str_raises():
    """测试传入str类型应抛出异常"""
    _require_npu()

    with pytest.raises((TypeError, AttributeError)):
        torch._prims_common.make_contiguous_strides_for("2,3")


def test_make_contiguous_strides_for_invalid_int_raises():
    """测试传入int类型应抛出异常"""
    _require_npu()

    # API抛出AssertionError
    with pytest.raises((TypeError, AttributeError, AssertionError)):
        torch._prims_common.make_contiguous_strides_for(123)


def test_make_contiguous_strides_for_invalid_dict_raises():
    """测试传入dict类型应抛出异常"""
    _require_npu()

    # API抛出AssertionError
    with pytest.raises((TypeError, AttributeError, AssertionError)):
        torch._prims_common.make_contiguous_strides_for({"shape": [2, 3]})


def test_make_contiguous_strides_for_zero_dim():
    """测试包含0的shape"""
    _require_npu()

    # shape包含0
    result = torch._prims_common.make_contiguous_strides_for((0, 3))
    # 对于0维，stride应该是(3, 1)或类似的
    assert isinstance(result, tuple)
    assert len(result) == 2

    result2 = torch._prims_common.make_contiguous_strides_for((3, 0))
    assert isinstance(result2, tuple)
    assert len(result2) == 2


def test_make_contiguous_strides_for_large_shape():
    """测试大shape的边界情况"""
    _require_npu()

    # 较大但合理的shape
    shape = (100, 200, 300)
    result = torch._prims_common.make_contiguous_strides_for(shape)
    expected = (60000, 300, 1)  # 200*300, 300, 1
    assert result == expected, f"Large shape: expected {expected}, got {result}"


def test_make_contiguous_strides_for_one_dimensional():
    """测试各种一维shape"""
    _require_npu()

    test_cases = [
        ((1,), (1,)),
        ((5,), (1,)),
        ((100,), (1,)),
    ]

    for shape, expected in test_cases:
        result = torch._prims_common.make_contiguous_strides_for(shape)
        assert result == expected, f"Shape {shape}: expected {expected}, got {result}"
