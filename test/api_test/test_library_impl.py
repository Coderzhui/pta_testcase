"""
测试目的：验证 `torch.library.impl` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.library.impl`
覆盖的入参维度：
- 参数传参与不传参：覆盖 decorator 形式注册和非法 qualname。
- 参数为 None / 非 None：覆盖 `lib=None` 默认路径。
- 枚举/多类型：覆盖 `PrivateUse1` 与 `CPU` dispatch type。
- 正常输入：覆盖自定义 op 在 NPU Tensor 上的分发。
- 异常输入：覆盖非法算子名。
- 边界值和等价类：覆盖单 Tensor 入参。
未覆盖项及原因：
- 未覆盖更复杂 schema 或多返回值注册场景。
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

def test_library_impl_registers_privateuse1_kernel_for_npu():
    _require_npu()

    namespace = "pta_test_library_impl"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("myop(Tensor x) -> Tensor")

    @torch.library.impl(f"{namespace}::myop", "PrivateUse1")
    def myop_privateuse1(x):
        return x + 2

    @torch.library.impl(f"{namespace}::myop", "CPU")
    def myop_cpu(x):
        return x + 1

    out = torch.ops.pta_test_library_impl.myop(torch.ones(2, device="npu"))
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "npu"
    assert tuple(out.shape) == (2,)


def test_library_impl_invalid_qualname_raises():
    _require_npu()

    with pytest.raises((ValueError, RuntimeError)):
        torch.library.impl("bad", "CPU", lambda x: x)
