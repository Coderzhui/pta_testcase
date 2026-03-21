# Test purpose: validate torch._logging.warning_once behavior on NPU-backed inputs,
# covering one-shot emission semantics, argument forwarding, cache separation, and
# predictable interface errors.
# API name: torch._logging.warning_once
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | logger_obj passed / not passed | yes | valid logger object and missing-argument error |
# | logger_obj valid / invalid | yes | logger object with `warning` method and non-object inputs |
# | message args boundary values | yes | empty args, string message, and NPU tensor message payload |
# | kwargs passed / not passed | yes | forwards kwargs unchanged |
# | repeated call / first call | yes | cached once-per-argument-bundle behavior |
# | distinct logger instances | yes | same message on different logger objects logs separately |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | real logging backend output | this test uses a dummy logger to isolate `warning_once` semantics and avoid global logging side effects. |
# | severity-level routing | `warning_once` only forwards to `logger.warning`; routing is owned by the logger implementation. |
# | backend-specific tensor math | the test only uses NPU tensors as message payloads, not for computation. |

import pytest
import torch
import torch_npu  # noqa: F401

import torch._logging as torch_logging


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


class DummyLogger:
    def __init__(self):
        self.calls = []

    def warning(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def clear_warning_once_cache():
    torch_logging.warning_once.cache_clear()
    yield
    torch_logging.warning_once.cache_clear()


@pytest.mark.parametrize("shape", [(), (0,), (2,)])
def test_warning_once_emits_once_per_unique_argument_bundle(shape):
    logger1 = DummyLogger()
    logger2 = DummyLogger()
    payload = torch.ones(shape, device="npu:0")

    torch_logging.warning_once(logger1, "payload=%s", payload, stacklevel=2)
    torch_logging.warning_once(logger1, "payload=%s", payload, stacklevel=2)
    torch_logging.warning_once(logger2, "payload=%s", payload, stacklevel=2)

    assert len(logger1.calls) == 1
    assert len(logger2.calls) == 1
    assert logger1.calls[0][0] == ("payload=%s", payload)
    assert logger1.calls[0][1] == {"stacklevel": 2}
    assert logger2.calls[0][0] == ("payload=%s", payload)
    assert logger2.calls[0][1] == {"stacklevel": 2}


def test_warning_once_forwards_empty_message_args_and_kwargs():
    logger = DummyLogger()

    torch_logging.warning_once(logger, stacklevel=3)

    assert logger.calls == [((), {"stacklevel": 3})]


def test_warning_once_missing_logger_obj_raises_type_error():
    with pytest.raises(TypeError):
        torch_logging.warning_once()


@pytest.mark.parametrize("bad_logger", [None, 1, "not-a-logger"])
def test_warning_once_rejects_non_logger_objects(bad_logger):
    with pytest.raises(AttributeError):
        torch_logging.warning_once(bad_logger, "message")
