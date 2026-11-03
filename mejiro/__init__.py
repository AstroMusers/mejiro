import logging as _logging
import os as _os

# On the first `import jax`, JAX's plugin discovery calls the CUDA plugin's
# initialize(), which fails (cuInit error 303) on GPU-less hosts and is logged at
# ERROR level even though JAX silently and correctly falls back to CPU. Setting
# JAX_PLATFORMS/JAX_PLATFORM_NAME does NOT suppress this. Raising the logger level
# before JAX is imported anywhere downstream silences the harmless traceback.
_logging.getLogger("jax._src.xla_bridge").setLevel(_logging.CRITICAL)

# Loading entries from the persistent JIT cache makes XLA's cpu_aot_loader log a
# huge ERROR-level "machine type ... doesn't match" message per kernel, because
# the stored compile features include tuning flags (+prefer-no-gather/-scatter)
# that are never host features. It is harmless. Level 3 shows only FATAL C++
# logs; must be set before jax is imported (jax itself setdefaults it to 1).
# setdefault so `TF_CPP_MIN_LOG_LEVEL=0` in the shell still works for debugging.
# Spawned workers inherit it through the environment.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__author__ = "Bryce Wedig"
__version__ = "3.0.0"
