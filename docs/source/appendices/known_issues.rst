Known Issues
============

Problems in ``mejiro``'s dependencies that affect ``mejiro``, with workarounds. Checked against the versions pinned in ``pyproject.toml``.

``pyHalo`` version 1.4.6 (the version ``mejiro`` pins) does not allow line-of-sight halos beyond redshift 4. This can be fixed by modifying the source code: change the maximum redshift from 4 to 8 in line 113 of ``pyHalo.Cosmology.cosmology.Cosmology._interp_kpc_per_asec()`` in ``pyHalo/Cosmology/cosmology.py``.
