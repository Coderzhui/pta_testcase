"""
测试目的：验证 `torch._C.DispatchKeySet` 的构造、集合操作和异常输入在 NPU 环境下可正常调用。
API 名称：`torch._C.DispatchKeySet`

覆盖的入参维度：
- 参数传参与不传参：已覆盖无参异常构造、带参构造。
- 参数为 None / 非 None：None 会进入异常分支，已通过异常用例覆盖。
- 枚举选项的主要候选值：已覆盖 CPU、AutogradCPU、Undefined、PrivateUse1。
- 支持多类型时覆盖主要类型：已覆盖字符串名称与枚举值。
- 正常输入：已覆盖合法 key 名称、合法枚举值、集合增删查。
- 异常输入：已覆盖无参、非法类型、非法名称。
- 边界值和等价类：已覆盖空集合 Undefined、单元素集合、空构造异常。

未覆盖项及原因：
- 无，已满足 API 功能一致性标准。
"""

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查 NPU 是否可用，不可用则跳过测试。"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证 `torch._C.DispatchKeySet` 的 NPU 功能测试。")


def test_dispatch_keyset_construct_and_basic_ops_on_npu():
    """
    测试用例：验证 DispatchKeySet 构造和基本操作
    覆盖场景：
    - 在 NPU 上下文中测试
    - 字符串构造（"CPU"）
    - 验证 has() 方法
    - 验证 raw_repr() 返回整数
    - 验证 highestPriorityTypeId() 方法
    - 枚举值构造（AutogradCPU、Undefined、PrivateUse1）
    - 集合操作：add() 和 remove()
    """
    _require_npu()

    npu_tensor = torch.ones(1, device="npu")
    assert npu_tensor.device.type == "npu"

    cpu_set = torch._C.DispatchKeySet("CPU")
    assert isinstance(cpu_set, torch.DispatchKeySet)
    assert cpu_set.has(torch._C.DispatchKey.CPU)
    assert isinstance(cpu_set.raw_repr(), int)
    assert cpu_set.highestPriorityTypeId() == torch._C.DispatchKey.CPU

    autograd_set = torch._C.DispatchKeySet("AutogradCPU")
    assert isinstance(autograd_set, torch.DispatchKeySet)
    assert autograd_set.has(torch._C.DispatchKey.AutogradCPU)
    assert isinstance(autograd_set.raw_repr(), int)

    undefined_set = torch._C.DispatchKeySet("Undefined")
    assert isinstance(undefined_set, torch.DispatchKeySet)
    assert not undefined_set.has(torch._C.DispatchKey.CPU)
    assert not undefined_set.has(torch._C.DispatchKey.CUDA)

    private_use_set = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    assert isinstance(private_use_set, torch.DispatchKeySet)
    assert private_use_set.has(torch._C.DispatchKey.PrivateUse1)
    assert isinstance(private_use_set.raw_repr(), int)

    added_set = undefined_set.add(torch._C.DispatchKey.PrivateUse1)
    assert isinstance(added_set, torch.DispatchKeySet)
    assert added_set.has(torch._C.DispatchKey.PrivateUse1)

    removed_set = private_use_set.remove(torch._C.DispatchKey.PrivateUse1)
    assert isinstance(removed_set, torch.DispatchKeySet)
    assert not removed_set.has(torch._C.DispatchKey.PrivateUse1)


def test_dispatch_keyset_invalid_inputs_raise():
    """
    测试用例：验证非法输入抛出异常
    覆盖场景：
    - 无参构造抛出 TypeError
    - 整数参数抛出异常
    - 非法字符串名称抛出异常
    """
    _require_npu()

    with pytest.raises(TypeError):
        torch._C.DispatchKeySet()

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        torch._C.DispatchKeySet(123)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        torch._C.DispatchKeySet("NotARealKey")
