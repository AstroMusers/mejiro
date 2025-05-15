Known Issues
############

``pysynphot`` version 2.0.0 uses the deprecated ``numpy`` method ``alltrue`` and will throw warnings. To avoid this, you can modify the source code: replace ``alltrue`` with ``all`` in lines 233 and 234 in ``pysynphot/spectrum.py``.