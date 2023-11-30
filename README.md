# Overview

Pipeline to generate realistic simulated Roman Space Telescope WFI images of galaxy-galaxy strong gravitational lenses built on [`lenstronomy`](https://github.com/lenstronomy/lenstronomy) and [Pandeia](https://outerspace.stsci.edu/display/PEN)

# Execution notes

## Environment

Packages requiring installation are:
* `conda`
  * `jupyter`
  * `lenstronomy`
  * `tqdm`
  * `photutils`
  * `acstools`
  * `astroquery`
* `pip`
  * Pandeia via `pip install pandeia.engine==3.0`
  * [`hydra`](https://hydra.cc/docs/intro/) via `pip install hydra-core`
  * `regions` via `pip install regions`

where my Miniconda installations are configured to default to the `conda-forge` channel over `defaults`.

## Configuration files

Every machine this code is run on should have a corresponding yaml configuration file in `config/machine` to specify
absolute filepaths to certain directories. Then, either update the
`defaults.machine` in `hydra/config.yaml` to whatever the machine's yaml file is named, or run code with argument
`machine=hpc` e.g. `python train.py machine=hpc` to override the default (`local`).
