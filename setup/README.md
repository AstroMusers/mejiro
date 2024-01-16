For versions <1.11.3 of `lenstronomy`, one change needs to be made to the local `lenstronomy` installation:

1. The file `nfw_core_truncated.py` needs to be added to generate the CDM subhalos using `pyHalo`. For me, the location
   is `/data/bwedig/.conda/envs/mejiro/lib/python3.10/site-packages/lenstronomy/LensModel/Profiles/nfw_core_truncated.py`.

Note that this file is included in `lenstronomy` version 1.11.3 and beyond.
