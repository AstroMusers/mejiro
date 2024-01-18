# Useful commands

* `ps u --sort -pcpu $(pgrep -vu root) | head -5`
* `pkill -9 -f script.py`

# Execution times

* 2992 models, num_pix=97, side=10.67, grid_oversample=3, num_samples=100000
  `01` on zenobia (40 cores), 00:00:10
  `02` on galileo (64 cores), 
  `03` on galileo (64 cores), 11:56:44; on uzay (64 cores), 13:16:50
  `04` on galileo (64 cores),

# Ideal state

* pip-installable
* functions as an add-on to `lenstronomy`
    * take `kwargs_model`, `kwargs_params` as input - or maybe unpack an ImageModel and prompt for redshifts
    * specify a dark matter substructure via `pyHalo`
    * specify some detector params, or just hand to a wrapper that has defaults and can directly provide a reasonable
      output
    * generate image
* examples
    1. basic: lenstronomy lens to image with all defaults. some simplest wrapper of everything: make sure all
       assumptions are stated in the notebook
    2. advanced: same lenstronomy lens to image with different options e.g. off-center PSF. expose the methods that
       comprise the wrapper in the previous example
    3. use the `plots` module to compare the two

# TODO

* tests, e.g. `pandeia.engine.pandeia_version()`, lenstronomy has a test
* automatically run Jupyter notebooks on commit via gh action as regression test