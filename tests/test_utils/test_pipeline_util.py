from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mejiro.utils import pipeline_util


def test_verify_jax_platform_reports_device_count(monkeypatch):
    logger = Mock()
    completed_process = SimpleNamespace(returncode=0, stdout='2\n', stderr='')
    run = Mock(return_value=completed_process)
    monkeypatch.setattr(pipeline_util.subprocess, 'run', run)

    pipeline_util.verify_jax_platform('cpu', logger)

    run.assert_called_once()
    assert run.call_args.kwargs['timeout'] == 120
    logger.info.assert_called_once_with("JAX platform 'cpu' ready (2 device(s))")


def test_verify_jax_platform_reports_probe_error(monkeypatch):
    logger = Mock()
    completed_process = SimpleNamespace(
        returncode=1,
        stdout='',
        stderr='first line\nbackend unavailable\n',
    )
    monkeypatch.setattr(
        pipeline_util.subprocess, 'run', Mock(return_value=completed_process)
    )

    with pytest.raises(RuntimeError, match='backend unavailable'):
        pipeline_util.verify_jax_platform('gpu', logger)