Structure
=========

``mejiro`` follows three import rules. ``lint-imports`` checks them in CI, using the contracts in ``pyproject.toml``. Imports inside functions count, but imports made through ``importlib`` can't be checked.

1. **Layers.** A module may import only from its own layer or a lower one, directly or indirectly. ``import mejiro`` is allowed everywhere, for finding the package's files. Every top-level module must belong to a layer. From highest to lowest:

   - ``pipeline``
   - ``exposure``, ``galaxy_galaxy``, ``lensed_supernova``, ``plots``, ``point_wfi``, ``strong_lens``, ``style``, ``synthetic_image``
   - ``instruments``
   - ``engines``, ``analysis``, ``cosmo``
   - ``utils``

2. **Independence.** The modules in ``mejiro.engines`` don't import each other, directly or indirectly, and neither do the modules in ``mejiro.instruments``. They may share their base classes in ``engine.py`` and ``instrument.py``.
3. **One backend per engine.** Each engine imports only its own backend out of ``galsim``, ``lenstronomy``, ``pandeia`` and ``stpsf``. The exception is ``pandeia_engine``, which also imports ``lenstronomy``.

The diagram below shows which ``mejiro`` modules import which. It's generated with ``pydeps`` every time the docs are built, so it always matches the code. Pipeline scripts are left out to keep it readable. Modules drawn as blue boxes are part of an import cycle.

.. image:: /_generated/mejiro.svg
    :alt: mejiro module dependency diagram
    :align: center