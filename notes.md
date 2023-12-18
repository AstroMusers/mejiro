# Useful commands

* `ps u --sort -pcpu $(pgrep -vu root) | head -10`
* `pkill -9 -f script.py`
* To suppress warnings in a script, `python3 -W ignore script.py`

# Execution times

* 3414 models, num_pix=97, side=10.67, grid_oversample=1, num_samples=10000
`01` on zenobia (40 cores), 00:00:07
`02` on jovian (64 cores), 00:37:09 (2572/3414 successful)
`03` on zenobia (40 cores), 
`04` on galileo (64 cores),  06:09:07

# TODO
* tests, e.g. `pandeia.engine.pandeia_version()`, lenstronomy has a test