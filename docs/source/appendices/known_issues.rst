Known Issues
############

``pysynphot`` version 2.0.0 uses the deprecated ``numpy`` method ``alltrue`` and will throw warnings. To avoid this, you can modify the source code: replace ``alltrue`` with ``all`` in lines 233 and 234 of ``pysynphot/spectrum.py``.

``speclite`` version 0.18 calls ``scipy.integrate.trapz`` and ``scipy.integrate.simps`` which were deprecated in favor of ``scipy.integrate.trapezoid`` and ``scipy.integrate.simpson``, respectively. The avoid this, you can modify the source code: replace ``trapz`` with ``trapezoid`` in line 271 and ``scipy.integrate.simps`` with ``scipy.integrate.simpson`` in line 272 of ``speclite/filters.py``.

``skypy`` version 0.5.3 will throw an error from ``speclite`` that wavelegnths do not cover the shifted response for HWO's HRI when running the ``SkyPyPipeline`` for HWO. Fix this by modifying the source code: add the following line after line 119 of ``skypy/utils/photometry.py``:

.. code::

    spectrum, wavelength = fs.pad_spectrum(spectrum, wavelength)

This pads the spectrum to allow the calculation of AB maggies for every redshift, spectrum, and filter.