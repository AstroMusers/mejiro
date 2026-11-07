Instruments
===========

``mejiro`` currently supports simulating the Roman Space Telescope ("Roman"), the James Webb Space Telescope (JWST), the Rubin Observatory's Legacy Survey of Space and Time (LSST), and the Habitable Worlds Observatory (HWO). ``mejiro``'s Roman imaging wraps ``GalSim`` and ``STPSF`` with instrument parameters from ``roman-technical-information``, its JWST imaging wraps ``GalSim`` and ``STPSF`` with instrument parameters from ``lenstronomy``, its LSST imaging wraps ``GalSim`` with Gaussian PSFs and instrument parameters from ``lenstronomy``, and its HWO imaging wraps ``GalSim`` with instrument parameters from ``syotools``.

.. note::
    ``mejiro``'s LSST and HWO support is currently under development.

For example, to simulate a *JWST* NIRCam exposure with GalSim, using one of the JWST PSFs that ship with ``mejiro``:

.. literalinclude:: ../examples/exposure_galsim_jwst.py
    :language: python

The other instruments work the same way; see :doc:`synthetic_image` and :doc:`engines` for *Roman*.

``instrument``
*******************

.. automodule:: mejiro.instruments.instrument
    :members:
    :undoc-members:
    :show-inheritance:

``roman``
*********

.. automodule:: mejiro.instruments.roman
    :members:
    :undoc-members:
    :show-inheritance:

``point_wfi``
*************

.. automodule:: mejiro.point_wfi
    :members:
    :undoc-members:
    :show-inheritance:

``jwst``
********

.. automodule:: mejiro.instruments.jwst
    :members:
    :undoc-members:
    :show-inheritance:

``lsst``
********

.. automodule:: mejiro.instruments.lsst
    :members:
    :undoc-members:
    :show-inheritance:

``hwo``
*******

.. automodule:: mejiro.instruments.hwo
    :members:
    :undoc-members:
    :show-inheritance: