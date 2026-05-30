import logging as _logging

# On the first `import jax`, JAX's plugin discovery calls the CUDA plugin's
# initialize(), which fails (cuInit error 303) on GPU-less hosts and is logged at
# ERROR level even though JAX silently and correctly falls back to CPU. Setting
# JAX_PLATFORMS/JAX_PLATFORM_NAME does NOT suppress this. Raising the logger level
# before JAX is imported anywhere downstream silences the harmless traceback.
_logging.getLogger("jax._src.xla_bridge").setLevel(_logging.CRITICAL)

__author__ = "Bryce Wedig"
__version__ = "3.0.0"
