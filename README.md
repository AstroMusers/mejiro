# Overview

`mejiro` ("MEH-ji-roe")[^1] is a pipeline to simulate Roman Space Telescope WFI images of galaxy-galaxy strong gravitational lenses
built on [`lenstronomy`](https://github.com/lenstronomy/lenstronomy), [`pyHalo`](https://github.com/dangilman/pyHalo), and [Pandeia](https://outerspace.stsci.edu/display/PEN).

# Setup

Create a conda environment:
```
conda env create --file environment.yml
```

Install Pandeia by following the instructions [here](https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+Installation). Version 3.1 is current (as of 2023-12-16) but `mejiro` uses v3.0. Make sure to download the Roman dataset for v3.0.

Then, follow the instructions in the `setup` directory to enable CDM subhalo generation with `pyHalo` on `lenstronomy`<=1.11.0 by modifying the local installation.

# Execution

First, duplicate any yaml file in `config/machine` and rename it, then update `repo_dir` and `data_dir` attributes to point to the directories where the local repository resides and output data should be written, respectively. Then, update pipeline parameters in `config/config.yaml` and ensure `defaults.machine` points to the name (without extension) of the yaml file created earlier.

The bash script `execute_pipeline.sh` will execute the pipeline end-to-end:
```
bash execute_pipeline.sh
```

[^1]: "mejiro" ([メジロ](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%B8%E3%83%AD) in Japanese) or "warbling white-eye" (*Zosterops japonicus*) is a small bird native to East Asia with a distinctive white ring around the eye, much like the shape of a strong lens.