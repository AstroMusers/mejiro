Known Issues
############

``pysynphot`` version 2.0.0 uses the deprecated ``numpy`` method ``alltrue`` and will throw warnings. To avoid this, you can modify the source code: replace ``alltrue`` with ``all`` in lines 233 and 234 of ``pysynphot/spectrum.py``.

``speclite`` version 0.18 calls ``scipy.integrate.trapz`` and ``scipy.integrate.simps`` which were deprecated in favor of ``scipy.integrate.trapezoid`` and ``scipy.integrate.simpson``, respectively. The avoid this, you can modify the source code: replace ``trapz`` with ``trapezoid`` in line 271 and ``scipy.integrate.simps`` with ``scipy.integrate.simpson`` in line 272 of ``speclite/filters.py``.