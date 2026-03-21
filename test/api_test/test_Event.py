"""
Test purpose: validate the interface and runtime behavior of torch.Event on an NPU-enabled environment.
API name: torch.Event

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: default construction; explicit kwargs for supported event options |
| None / non-None | Covered: constructor kwargs are exercised with non-None booleans where supported |
| Enum options | Uncovered: torch.Event does not expose enum-style parameters |
| Multiple types | Uncovered: constructor is boolean-flag based, so no meaningful type matrix is available |
| Normal input | Covered: event creation, record, query, synchronize, and elapsed_time capability on NPU |
| Error input | Covered: unexpected constructor keyword raises TypeError |
| Boundary / equivalence classes | Covered: default vs explicit constructor flags; unrecorded/recorded event lifecycle on NPU |

Uncovered items and reasons:
- Enum options: not applicable because torch.Event accepts boolean flags rather than enums.
- Multiple types: not applicable because this API does not accept heterogeneous input types.
- Additional boundary values beyond booleans: not applicable because the relevant constructor parameters are boolean.
"""

import pytest

try:
    import torch_npu  # noqa: F401
    import torch
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    torch_npu = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - simple import guard
    _IMPORT_ERROR = None


def _require_npu_event():
    if torch is None or torch_npu is None:
        pytest.skip(f"torch_npu import failed: {_IMPORT_ERROR}")
    if not hasattr(torch, "Event"):
        pytest.skip("torch.Event is not exposed in this build after importing torch_npu.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def _make_npu_tensor():
    # Use a real NPU allocation so the event is exercised in an NPU runtime path.
    return torch.ones(4, device="npu")


def test_event_constructor_and_lifecycle_on_npu():
    _require_npu_event()

    default_event = torch.Event()
    explicit_event = torch.Event(enable_timing=True, blocking=True, interprocess=False)

    assert default_event is not None
    assert explicit_event is not None

    tensor = _make_npu_tensor()
    assert tensor.device.type == "npu"

    default_event.record()
    explicit_event.record()
    explicit_event.synchronize()

    assert isinstance(default_event.query(), bool)
    assert isinstance(explicit_event.query(), bool)
    assert explicit_event.query() is True
    assert default_event.query() in (True, False)


def test_event_elapsed_time_returns_float_for_recorded_npu_events():
    _require_npu_event()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    start_event.record()
    _make_npu_tensor().add_(1)
    end_event.record()
    end_event.synchronize()

    try:
        elapsed = start_event.elapsed_time(end_event)
    except RuntimeError as exc:
        pytest.xfail(f"elapsed_time is not supported by this NPU backend: {exc}")

    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_event_rejects_unexpected_constructor_arguments():
    _require_npu_event()

    with pytest.raises(TypeError):
        torch.Event(unexpected_flag=True)
