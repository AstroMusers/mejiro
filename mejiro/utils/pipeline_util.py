import importlib
import subprocess
import sys


def verify_jax_platform(jax_platform, logger):
    """Verify a JAX platform without locking the orchestrator's backend."""
    code = (
        "import logging;"
        "logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL);"
        "import os;"
        f"os.environ['JAX_PLATFORM_NAME']={jax_platform!r};"
        f"os.environ['JAX_PLATFORMS']={jax_platform!r};"
        "import jax;"
        f"devices=jax.devices({jax_platform!r});"
        "print(len(devices))"
    )
    try:
        process = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f'Timed out probing jax platform {jax_platform!r}.'
        ) from error

    if process.returncode != 0:
        error_detail = (
            process.stderr.strip().splitlines()[-1]
            if process.stderr.strip()
            else '(no stderr)'
        )
        raise RuntimeError(
            f'JAX platform {jax_platform!r} unavailable on this host: {error_detail}'
        )
    logger.info(
        f'JAX platform {jax_platform!r} ready ({process.stdout.strip()} device(s))'
    )


def initialize_instrument_class(instrument_name):
    base_module_path = "mejiro.instruments"
    class_map = {
        "hwo": "HWO",
        "roman": "Roman"
    }

    if instrument_name.lower() not in class_map:
        raise ValueError(f"Unknown instrument: {instrument_name}")

    module_path = f"{base_module_path}.{instrument_name.lower()}"
    module = importlib.import_module(module_path)
    class_name = class_map[instrument_name.lower()]
    cls = getattr(module, class_name)
    instance = cls()
    
    return instance
