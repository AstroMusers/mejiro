Engines
#######

``mejiro`` wraps various image simulation packages to provide a consistent interface for simulating images. Each wrapped package is referred to as an "engine." The following engines are currently supported:

- GalSim: *Roman*, *HWO*
- Lenstronomy: *Roman*
- Pandeia: *Roman*
- Romanisim: *Roman*
- STPSF (PSFs only): *Roman*

An engine is specified when generating a simulated exposure, i.e., an instance of ``mejiro.exposure.Exposure``. The engine is specified in the ``engine`` keyword argument, and any specific parameters passed to the engine are specified in the ``engine_params`` dictionary. For example, to simulate a *Roman* image using GalSim with ,

.. code-block:: python

    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure

    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                    instrument=Roman(),
                                    band='F129',
                                    fov_arcsec=5)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim')

To see the available options for each engine, check the default parameters using ``defaults(instrument_name)``. For example, the default parameters for simulating *Roman* images with GalSim are:

.. code-block:: python

    from mejiro.engines.galsim_engine import defaults
    from pprint import pprint

    pprint(defaults('Roman'))

::

    {
        'rng_seed': 42,
        'sky_background': True,
        'detector_effects': True,
        'poisson_noise': True,
        'reciprocity_failure': True,
        'dark_noise': True,
        'nonlinearity': True,
        'ipc': True,
        'read_noise': True,
    }


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

.. ``romanisim_engine``
.. ********************

.. .. automodule:: mejiro.engines.romanisim_engine
..     :members:
..     :undoc-members:
..     :show-inheritance:

``stpsf_engine``
******************

.. automodule:: mejiro.engines.stpsf_engine
    :members:
    :undoc-members:
    :show-inheritance:
