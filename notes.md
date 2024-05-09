# Useful commands

* `ps u --sort -pcpu $(pgrep -vu root) | head -5`
* `pkill -9 -f script.py`
* `jupyter nbconvert --to python pipeline.ipynb --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input_prompt=True`

# Environment management

* `requirements.txt`: `pip list --format=freeze > requirements.txt`

# Execution times

* Pandeia: 2992 models, num_pix=97, side=10.67, grid_oversample=3, num_samples=100000
  `01` on galileo (64 cores), 00:00:12 (0.004s/image)
  `02` on galileo (64 cores), 00:33:23 (0.67s/image)
  `03` on galileo (64 cores), ; on uzay (64 cores), 13:16:50 (15.98s/image)
  `04` on galileo (64 cores),

* GalSim: 259 models, num_pix=96, side=10.56, grid_oversample=3
  `01` on galileo (64 cores), 00:00:06 (0.02s/image)
  `02` on galileo (64 cores), 00:02:50 (0.66s/image)
  `03` on galileo (64 cores), 01:23:49 (19.42s/image)
  `04` on galileo (64 cores), 00:05:37 (19.42s/image)
  `05` on galileo (64 cores), 00:00:06 (19.42s/image)

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
