"""
测试目的：验证 `torch.library.Library` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.library.Library`
覆盖的入参维度：
- 参数传参与不传参：覆盖 `DEF` / `FRAGMENT` 构造和非法 kind。
- 参数为 None / 非 None：覆盖空 dispatch key 与显式 `PrivateUse1`。
- 枚举/多类型：覆盖不同 `kind`。
- 正常输入：覆盖建库、定义 op 和 NPU 分发。
- 异常输入：覆盖非法 kind。
- 边界值和等价类：覆盖最小 schema。
未覆盖项及原因：
- 未覆盖库生命周期管理的更复杂冲突场景。
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

def test_library_construct_and_define_for_npu_dispatch():
    _require_npu()

    namespace = "pta_test_library_ctor"
    lib = torch.library.Library(namespace, "DEF")
    frag = torch.library.Library(namespace, "FRAGMENT", "PrivateUse1")
    lib.define("myop(Tensor x) -> Tensor")

    @torch.library.impl(f"{namespace}::myop", "PrivateUse1")
    def myop_privateuse1(x):
        return x.clone()

    out = torch.ops.pta_test_library_ctor.myop(torch.ones(1, device="npu"))
    assert isinstance(lib, torch.library.Library)
    assert isinstance(frag, torch.library.Library)
    assert out.device.type == "npu"


def test_library_invalid_kind_raises():
    _require_npu()

    with pytest.raises(ValueError):
        torch.library.Library("pta_bad_kind", "BAD")
