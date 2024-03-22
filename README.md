> :warning: **Currently under active development**

# Overview

`mejiro` ("MEH-ji-roe")[^1] is a pipeline to simulate Roman Space Telescope WFI images of galaxy-galaxy strong
gravitational lenses
built
on [`lenstronomy`](https://github.com/lenstronomy/lenstronomy), [`pyHalo`](https://github.com/dangilman/pyHalo), [GalSim](https://github.com/GalSim-developers/GalSim),
and [Pandeia](https://outerspace.stsci.edu/display/PEN).

# Setup

Create a conda environment:

```
conda env create -f environment.yml
```

## Optional setup: Pandeia

Install Pandeia by following the
instructions [here](https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation). Version 3.1 is current (as of
2023-12-16) but `mejiro` uses v3.0. Make sure to download the Roman dataset for v3.0.

# Getting Started

First, duplicate any yaml file in the `config/machine` directory and rename it, then update `repo_dir` and `data_dir`
attributes to point to the directories where the local repository resides and output data should be written,
respectively. Then modify the `defaults.machine` attribute in `config/config.yaml` to the name (without extension) of
the yaml file created earlier.

Then, execute the Jupyter notebooks in the `examples` directory in order.

# Pipeline Execution

Update pipeline parameters in `config/config.yaml` and ensure `defaults.machine` points to the name (without extension)
of the yaml file created earlier. The bash script `execute_pipeline.sh` will execute the pipeline end-to-end:

```
time bash execute_pipeline.sh
```

[^1]: "mejiro" ([メジロ](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%B8%E3%83%AD) in Japanese) or "warbling
white-eye" (*Zosterops japonicus*) is a small bird native to East Asia with a distinctive white ring around the eye,
much like the shape of a strong lens.
