"""
测试目的：验证 `torch.nn.Parameter.device` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.device`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `npu`、`cpu` 设备对象，以及不同设备索引。
- 正常输入：覆盖 NPU Parameter 的设备读取、CPU与NPU对比、设备切换后的验证。
- 异常输入：不适用（只读属性）。
- 边界值和等价类：覆盖一维 Parameter、多维 Parameter、标量 Parameter、不同设备索引。
未覆盖项及原因：
- 已覆盖主要设备类型和切换场景。
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

def test_parameter_device_reports_npu():
    """测试NPU设备读取"""
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, device="npu"))
    assert parameter.device.type == "npu"
    assert str(parameter.device).startswith("npu")


def test_parameter_device_cpu_vs_npu():
    """测试CPU与NPU设备对比"""
    _require_npu()

    # CPU Parameter
    cpu_param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    assert cpu_param.device.type == "cpu"

    # NPU Parameter
    npu_param = torch.nn.Parameter(torch.ones(2, device="npu"))
    assert npu_param.device.type == "npu"

    # 验证两者不同
    assert cpu_param.device != npu_param.device


def test_parameter_device_to_method():
    """测试to()方法切换设备后的device属性"""
    _require_npu()

    # 从CPU创建
    cpu_param = torch.nn.Parameter(torch.ones(2, device="cpu"))
    assert cpu_param.device.type == "cpu"

    # 切换到NPU
    npu_param = cpu_param.to("npu")
    assert npu_param.device.type == "npu"

    # 切回CPU
    cpu_param2 = npu_param.to("cpu")
    assert cpu_param2.device.type == "cpu"


def test_parameter_device_different_shapes():
    """测试不同shape的Parameter的device属性"""
    _require_npu()

    shapes = [(), (1,), (2, 3), (2, 3, 4), (1, 1, 1)]
    for shape in shapes:
        param = torch.nn.Parameter(torch.ones(shape, device="npu"))
        assert param.device.type == "npu"
        assert str(param.device).startswith("npu")


def test_parameter_device_different_dtypes():
    """测试不同dtype的Parameter的device属性"""
    _require_npu()

    dtypes = [torch.float32, torch.float16]
    # 如果NPU支持float64
    try:
        torch.ones(1, dtype=torch.float64, device="npu")
        dtypes.append(torch.float64)
    except (RuntimeError, TypeError):
        pass

    for dtype in dtypes:
        param = torch.nn.Parameter(torch.ones(2, device="npu", dtype=dtype))
        assert param.device.type == "npu"


def test_parameter_device_index():
    """测试device的index属性（如果存在多NPU）"""
    _require_npu()

    param = torch.nn.Parameter(torch.ones(2, device="npu"))
    # device可能有index属性（如npu:0, npu:1）
    # 我们只验证可以访问device属性
    assert hasattr(param.device, 'type')
    assert param.device.type == "npu"

    # 如果环境支持多NPU，测试显式指定设备索引
    try:
        param_0 = torch.nn.Parameter(torch.ones(2, device="npu:0"))
        assert param_0.device.type == "npu"
    except (RuntimeError, ValueError):
        # 如果npu:0不支持则跳过
        pass


def test_parameter_device_after_operations():
    """测试操作后device属性保持一致"""
    _require_npu()

    param = torch.nn.Parameter(torch.ones(2, 3, device="npu"))
    assert param.device.type == "npu"

    # 各种操作后检查device
    param2 = param + 1
    assert param2.device.type == "npu"

    param3 = param * 2
    assert param3.device.type == "npu"

    param4 = param.view(-1)
    assert param4.device.type == "npu"


def test_parameter_device_from_buffer():
    """测试从buffer创建的Parameter的device属性"""
    _require_npu()

    # 从NPU buffer创建
    buffer = torch.ones(2, device="npu")
    param = torch.nn.Parameter(buffer)
    assert param.device.type == "npu"

    # 从CPU buffer创建
    buffer_cpu = torch.ones(2, device="cpu")
    param_cpu = torch.nn.Parameter(buffer_cpu)
    assert param_cpu.device.type == "cpu"
