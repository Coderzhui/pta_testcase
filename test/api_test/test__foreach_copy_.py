"""
测试目的：验证 `torch._foreach_copy_` 的功能行为，执行张量列表的复制操作。
API 名称：`torch._foreach_copy_`
覆盖的入参维度：
- 参数传参与不传参：覆盖传入self和src参数
- 参数为 None / 非 None：覆盖各种tensor列表
- 枚举/多类型：覆盖不同dtype、device的tensor
- 正常输入：覆盖tensor列表的复制操作
- 异常输入：覆盖shape不匹配、类型不匹配、空列表等
- 边界值和等价类：覆盖单元素列表、多元素列表、空列表
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_foreach_copy_basic():
    """测试基本的foreach_copy_操作"""
    _require_npu()

    # 创建源张量列表和目标张量列表
    src = [torch.ones(3, device="npu"), torch.ones(4, device="npu")]
    self_tensors = [torch.zeros(3, device="npu"), torch.zeros(4, device="npu")]

    # 执行复制
    result = torch._foreach_copy_(self_tensors, src)

    # 验证结果
    assert len(result) == 2
    assert torch.allclose(result[0], torch.ones(3, device="npu"))
    assert torch.allclose(result[1], torch.ones(4, device="npu"))
    # 验证原地修改
    assert torch.allclose(self_tensors[0], torch.ones(3, device="npu"))


def test_foreach_copy_single_element():
    """测试单元素列表"""
    _require_npu()

    src = [torch.ones(5, device="npu")]
    self_tensors = [torch.zeros(5, device="npu")]

    result = torch._foreach_copy_(self_tensors, src)

    assert len(result) == 1
    assert torch.allclose(result[0], torch.ones(5, device="npu"))


def test_foreach_copy_different_shapes():
    """测试不同shape的tensor"""
    _require_npu()

    shapes = [(2,), (2, 3), (2, 3, 4)]
    src = [torch.ones(shape, device="npu") for shape in shapes]
    self_tensors = [torch.zeros(shape, device="npu") for shape in shapes]

    result = torch._foreach_copy_(self_tensors, src)

    for i, shape in enumerate(shapes):
        assert result[i].shape == shape
        assert torch.allclose(result[i], torch.ones(shape, device="npu"))


def test_foreach_copy_different_dtypes():
    """测试不同dtype的tensor"""
    _require_npu()

    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        src = [torch.ones(3, device="npu", dtype=dtype)]
        self_tensors = [torch.zeros(3, device="npu", dtype=dtype)]

        result = torch._foreach_copy_(self_tensors, src)
        assert result[0].dtype == dtype
        assert torch.allclose(result[0], torch.ones(3, device="npu", dtype=dtype))


def test_foreach_copy_shape_mismatch_raises():
    """测试shape不匹配应抛出异常"""
    _require_npu()

    src = [torch.ones(3, device="npu")]
    self_tensors = [torch.zeros(4, device="npu")]  # 不同shape

    with pytest.raises(RuntimeError):
        torch._foreach_copy_(self_tensors, src)


def test_foreach_copy_empty_list_raises():
    """测试空列表应抛出异常"""
    _require_npu()

    src = []
    self_tensors = []

    with pytest.raises(RuntimeError):
        torch._foreach_copy_(self_tensors, src)


def test_foreach_copy_different_length_raises():
    """测试长度不匹配的列表应抛出异常"""
    _require_npu()

    src = [torch.ones(3, device="npu"), torch.ones(4, device="npu")]
    self_tensors = [torch.zeros(3, device="npu")]  # 长度不同

    with pytest.raises(RuntimeError):
        torch._foreach_copy_(self_tensors, src)


def test_foreach_copy_non_tensor_raises():
    """测试传入非tensor应抛出异常"""
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        torch._foreach_copy_(["not_tensor"], ["not_tensor"])


def test_foreach_copy_returns_list():
    """测试返回值类型为list"""
    _require_npu()

    src = [torch.ones(3, device="npu")]
    self_tensors = [torch.zeros(3, device="npu")]

    result = torch._foreach_copy_(self_tensors, src)

    assert isinstance(result, list)
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_foreach_copy_with_grad():
    """测试带梯度的tensor（需要在非叶子节点上使用）"""
    _require_npu()

    # 创建叶子节点
    src_leaf = torch.ones(3, device="npu", requires_grad=True)
    # 创建非叶子节点作为目标
    self_tensors = [src_leaf + 0]  # +0 操作使其成为非叶子节点

    src = [torch.ones(3, device="npu")]

    # 现在可以执行原地操作
    result = torch._foreach_copy_(self_tensors, src)

    assert torch.allclose(result[0], torch.ones(3, device="npu"))
