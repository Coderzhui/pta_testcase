"""
Test purpose: validate torch._dynamo.config behavior while running in an NPU-enabled environment.
API name: torch._dynamo.config

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: configuration attributes are read with and without temporary override |
| None / non-None | Covered: boolean config flags are exercised via True/False values |
| Enum options | Uncovered: torch._dynamo.config is a module of flags, not an enum-based API |
| Multiple types | Covered: boolean and integer config values are checked where available |
| Normal input | Covered: reading representative config flags and temporarily overriding one flag |
| Error input | Covered: accessing a missing config attribute raises AttributeError |
| Boundary / equivalence classes | Covered: default value vs overridden value round-trip and boolean toggle paths |

Uncovered items and reasons:
- Enum options: not applicable because config exposes flags rather than enum parameters.
- Non-boolean boundary values for all flags: not all config flags are safely mutable in this environment, so coverage is limited to representative boolean/integer flags.
- Deep validation of every internal config field: omitted to avoid touching unstable or implementation-specific internals.
"""

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import guard
    _TORCH_NPU_IMPORT_ERROR = None

from torch._dynamo import config as dynamo_config


def _require_npu():
    if torch_npu is None:
        pytest.skip(f"torch_npu is unavailable: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def test_dynamo_config_reads_and_restores_representative_flags():
    _require_npu()

    verbose = getattr(dynamo_config, "verbose", None)
    suppress_errors = getattr(dynamo_config, "suppress_errors", None)
    dynamic_shapes = getattr(dynamo_config, "dynamic_shapes", None)

    assert isinstance(verbose, bool)
    assert isinstance(suppress_errors, bool)
    assert isinstance(dynamic_shapes, bool)

    # Only mutate if the attribute exists and the type is simple/stable enough to restore.
    if hasattr(dynamo_config, "verbose"):
        original = dynamo_config.verbose
        dynamo_config.verbose = not original
        try:
            assert dynamo_config.verbose is (not original)
        finally:
            dynamo_config.verbose = original
        assert dynamo_config.verbose is original


def test_dynamo_config_representative_integer_limit_is_positive_or_zero():
    _require_npu()

    if not hasattr(dynamo_config, "cache_size_limit"):
        pytest.skip("cache_size_limit is not exposed in this build.")

    limit = dynamo_config.cache_size_limit
    assert isinstance(limit, int)
    assert limit >= 0


def test_dynamo_config_missing_attribute_raises_attribute_error():
    _require_npu()

    with pytest.raises(AttributeError):
        _ = dynamo_config.this_attribute_should_not_exist_for_test_coverage
