# Useful commands

* `ps u --sort -pcpu $(pgrep -vu root) | head -10`
* `pkill -9 -f script.py`
* To suppress warnings in a script, `python3 -W ignore script.py`

# Execution times

* x models, num_pix=97, side=10.67, grid_oversample=3, num_samples=10000
`01` on galileo (64 cores), 00:00:20
`02` on galileo (64 cores), 00:33:44
`03` on galileo (64 cores), 11:56:44
`04` on galileo (64 cores), 

# TODO
* tests, e.g. `pandeia.engine.pandeia_version()`, lenstronomy has a test