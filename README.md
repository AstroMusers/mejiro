> :warning: **Currently under active development**

# Overview

`mejiro` ("MEH-ji-roe")[^1] is a pipeline to simulate Roman Space Telescope Wide Field Instrument images of galaxy-galaxy strong
gravitational lenses
built
on [`lenstronomy`](https://github.com/lenstronomy/lenstronomy), [`pyHalo`](https://github.com/dangilman/pyHalo), [GalSim](https://github.com/GalSim-developers/GalSim),
and [Pandeia](https://outerspace.stsci.edu/display/PEN).

# Setup

Create a conda environment:

```
conda env create -f environment.yml
```

Install `SLSim`:

```
git clone https://github.com/LSST-strong-lensing/slsim.git
cd slsim
pip install -e .
```

## Optional setup: Pandeia

Install Pandeia (v3.1) by following the
instructions [here](https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation).

# Getting Started

1. Duplicate any yaml file in the `config/machine` directory and rename it, then update `repo_dir` and `data_dir`
attributes to point to the directories where the local repository resides and output data should be written,
respectively. 
2. Copy `setup/config.yaml` to `config/config.yaml`
3. Modify the `defaults.machine` attribute in the file you just copied (`config/config.yaml`) to the name (without extension) of the yaml file created in step 1.

Then, execute the Jupyter notebooks in the `examples` directory in order.

Note that this `hydra`-based configuration will be changed soon.

# Pipeline Execution

Update pipeline parameters in `config/config.yaml` and ensure `defaults.machine` points to the name (without extension)
of the yaml file created earlier. The bash script `execute_pipeline.sh` will execute the pipeline end-to-end:

```
time bash execute_pipeline.sh
```

[^1]: "mejiro" ([メジロ](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%B8%E3%83%AD) in Japanese) or "warbling
white-eye" (*Zosterops japonicus*) is a small bird native to East Asia with a distinctive white ring around the eye,
much like the shape of a strong lens.
