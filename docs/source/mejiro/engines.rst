Engines
=======

``mejiro`` wraps various image simulation packages to provide a consistent interface for simulating images. Each wrapped package is referred to as an "engine." The following engines are currently supported:

- GalSim: *Roman*, *JWST*, *LSST*, *HWO*
- STPSF (PSFs only): *Roman*

`romanisim <https://github.com/spacetelescope/romanisim>`_ is not available as an ``Exposure`` engine. Instead, *Roman* detector simulation with romanisim runs as pipeline step :mod:`mejiro.pipeline._05_romanisim`, selected by setting ``imaging.engine: romanisim`` in the pipeline config. That key only chooses the step 05 simulator: :mod:`mejiro.pipeline.calculate_snrs` always rebuilds its SNR exposures with GalSim, using ``imaging.engine_params``. romanisim doesn't read ``imaging.engine_params``, so it must hold GalSim parameters even when ``imaging.engine`` is ``romanisim``.

An engine is specified when generating a simulated exposure, i.e., an instance of ``mejiro.exposure.Exposure``. The engine is specified in the ``engine`` keyword argument, and any specific parameters passed to the engine are specified in the ``engine_params`` dictionary. For example, to simulate a *Roman* image using GalSim:

.. literalinclude:: ../examples/exposure_galsim_roman.py
    :language: python

To see the available options for each engine, check the default parameters using ``GalSimEngine.defaults(instrument_name)``. For example, the default parameters for simulating *Roman* images with GalSim are:

.. literalinclude:: ../examples/galsim_defaults.py
    :language: python

which prints:

.. literalinclude:: ../examples/galsim_defaults.txt
    :language: text


``engine``
*****************

.. automodule:: mejiro.engines.engine
    :members:
    :undoc-members:
    :show-inheritance:

``galsim_engine``
*****************

.. automodule:: mejiro.engines.galsim_engine
    :members:
    :undoc-members:
    :show-inheritance:

.. ``lenstronomy_engine``
.. **********************

.. .. automodule:: mejiro.engines.lenstronomy_engine
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. ``pandeia_engine``
.. ******************

.. .. automodule:: mejiro.engines.pandeia_engine
..     :members:
..     :undoc-members:
..     :show-inheritance:

``stpsf_engine``
******************

.. automodule:: mejiro.engines.stpsf_engine
    :members:
    :undoc-members:
    :show-inheritance:
