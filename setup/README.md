For version 1.11.0 of `lenstronomy`, a few changes need to be made to the local `lenstronomy` installation:
1. The file `nfw_core_truncated.py` needs to be added to generate the CDM subhalos using `pyHalo`. For me, the location
is `/data/bwedig/.conda/envs/mejiro/lib/python3.10/site-packages/lenstronomy/LensModel/Profiles/nfw_core_truncated.py`.
2. The method `magnitude2amplitude()` needs to be added to the `data_util.py` file, along with its required `import copy` statement. For me, the location is `/data/bwedig/.conda/envs/mejiro/lib/python3.10/site-packages/lenstronomy/Util/data_util.py`.
