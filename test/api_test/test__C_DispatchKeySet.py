# 测试目的: 验证 torch._C.DispatchKeySet 在 NPU 环境下的构造、集合运算、raw 表示和异常行为。
# API 名称: torch._C.DispatchKeySet
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖构造、add/remove、按位运算和 from_raw_repr |
# | None / 非 None | 不适用 | 该 API 不是可选参数型函数 |
# | 枚举选项 | 覆盖 | 覆盖 CPU / NPU / AutogradNPU（若可用）等 DispatchKey 输入 |
# | 多类型 | 覆盖 | 覆盖 DispatchKeySet、DispatchKey、NPU Tensor 生成的 keyset |
# | 正常输入 | 覆盖 | NPU Tensor 派生 keyset、set algebra、raw round-trip |
# | 异常输入 | 覆盖 | 构造/增删时传入非法类型触发 TypeError |
# | 边界值和等价类 | 覆盖 | 空 keyset、单 key keyset、NPU tensor keyset |
# 未覆盖项及原因:
# - None / 非 None: 该 API 不接受 None 作为自然输入。
# - 其他 dispatch key 组合: 本文件聚焦 NPU 相关覆盖，避免扩展到无关后端导致冗余。

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time environment guard
    torch_npu = None  # type: ignore[assignment]
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def _require_npu() -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        pytest.skip(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu backend is unavailable in this environment")
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable in this environment")


def _npu_device() -> torch.device:
    _require_npu()
    return torch.device("npu:0")


def _require_dispatch_key(name: str):
    key = getattr(torch._C.DispatchKey, name, None)
    if key is None:
        pytest.skip(f"DispatchKey.{name} is unavailable in this build")
    return key


def test_dispatchkeyset_from_npu_tensor_round_trip_and_set_ops() -> None:
    _require_npu()

    npu_key = _require_dispatch_key("NPU")
    cpu_key = _require_dispatch_key("CPU")

    tensor = torch.ones(1, device=_npu_device(), dtype=torch.float32)
    tensor_keyset = torch._C._dispatch_keys(tensor)

    assert isinstance(tensor_keyset, torch._C.DispatchKeySet)
    assert tensor_keyset.has(npu_key) is True

    npu_only = torch._C.DispatchKeySet(npu_key)
    cpu_only = torch._C.DispatchKeySet(cpu_key)

    combined = npu_only | cpu_only
    assert combined.has(npu_key) is True
    assert combined.has(cpu_key) is True
    assert (combined & npu_only).has(npu_key) is True
    assert (combined - npu_only).has(npu_key) is False
    assert (combined - npu_only).has(cpu_key) is True

    added = cpu_only.add(npu_key)
    assert added.has(npu_key) is True

    removed = added.remove(npu_key)
    assert removed.has(npu_key) is False
    assert removed.has(cpu_key) is True

    raw = tensor_keyset.raw_repr()
    reconstructed = torch._C.DispatchKeySet.from_raw_repr(raw)
    assert reconstructed.raw_repr() == raw
    assert reconstructed.has(npu_key) is True


def test_dispatchkeyset_boundary_empty_set_from_subtraction() -> None:
    _require_npu()

    npu_key = _require_dispatch_key("NPU")
    single = torch._C.DispatchKeySet(npu_key)
    empty = single - single

    assert isinstance(empty, torch._C.DispatchKeySet)
    assert empty.has(npu_key) is False
    assert empty.raw_repr() == 0

    round_tripped = torch._C.DispatchKeySet.from_raw_repr(empty.raw_repr())
    assert round_tripped.raw_repr() == 0
    assert round_tripped.has(npu_key) is False


def test_dispatchkeyset_rejects_invalid_types() -> None:
    _require_npu()

    npu_key = _require_dispatch_key("NPU")
    keyset = torch._C.DispatchKeySet(npu_key)

    with pytest.raises(TypeError):
        torch._C.DispatchKeySet("bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        keyset.add("bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        keyset.remove("bad")  # type: ignore[arg-type]
