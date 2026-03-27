"""
测试目的：验证 `torch._C._get_accelerator` 的功能行为，返回当前加速器设备。
API 名称：`torch._C._get_accelerator`
覆盖的入参维度：
- 参数传参与不传参：覆盖无参调用、传参调用
- 参数为 None / 非 None：覆盖 None、True、False 等多种输入
- 枚举/多类型：覆盖 bool、None 等类型
- 正常输入：覆盖各种合法参数组合
- 异常输入：覆盖非法类型参数
- 边界值和等价类：覆盖 None、True、False 等价类
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查NPU是否可用"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_get_accelerator_no_args():
    """测试无参调用返回加速器设备"""
    _require_npu()

    result = torch._C._get_accelerator()
    assert isinstance(result, torch.device)
    assert result.type == "npu"


def test_get_accelerator_with_true():
    """测试传入True参数"""
    _require_npu()

    result = torch._C._get_accelerator(True)
    assert isinstance(result, torch.device)
    assert result.type == "npu"


def test_get_accelerator_with_false():
    """测试传入False参数"""
    _require_npu()

    result = torch._C._get_accelerator(False)
    assert isinstance(result, torch.device)
    assert result.type == "npu"


def test_get_accelerator_with_none():
    """测试传入None参数"""
    _require_npu()

    result = torch._C._get_accelerator(None)
    assert isinstance(result, torch.device)
    assert result.type == "npu"


def test_get_accelerator_return_type():
    """测试返回值类型为torch.device"""
    _require_npu()

    result = torch._C._get_accelerator()
    assert isinstance(result, torch.device)
    # 验证device属性
    assert hasattr(result, 'type')
    assert hasattr(result, 'index')


def test_get_accelerator_with_int():
    """测试传入int类型的行为（API接受int并转换为bool）"""
    _require_npu()

    # API接受int参数，0转为False，非0转为True
    result = torch._C._get_accelerator(123)
    assert isinstance(result, torch.device)
    assert result.type == "npu"

    result_zero = torch._C._get_accelerator(0)
    assert isinstance(result_zero, torch.device)
    assert result_zero.type == "npu"


def test_get_accelerator_with_str_raises():
    """测试传入str类型应抛出异常"""
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        torch._C._get_accelerator("npu")


def test_get_accelerator_with_list_raises():
    """测试传入list类型应抛出异常"""
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        torch._C._get_accelerator([True])


def test_get_accelerator_consistency():
    """测试多次调用返回一致性"""
    _require_npu()

    result1 = torch._C._get_accelerator()
    result2 = torch._C._get_accelerator(True)
    result3 = torch._C._get_accelerator(False)
    result4 = torch._C._get_accelerator(None)

    # 验证所有调用返回相同的设备类型
    assert result1.type == result2.type == result3.type == result4.type
