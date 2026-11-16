# Running on the Roman Research Nexus: what's unverified

Status: open in mejiro; the RRN page and `lens_search_hlwas.ipynb` were written from the Nexus's public docs and haven't been run on the Nexus yet.
Last verified: 2026-11-15, without Nexus access. Locally (romanisim 0.14.0, romancal 1.1.0, asdf-astropy 0.11.0, CRDS context `roman_0078.pmap`), `pytest --run-crds` passed and the notebook ran end to end on one filter (F158, skycell `011m46x56y38`).

`docs/source/usage/roman_research_nexus.rst` relies on the following. Check each on the first
run there, fix the page (and this note) where it's wrong, then mark this note fixed or delete it.

- **Environment variables.** The page says the Nexus sets `CRDS_PATH`, `CRDS_SERVER_URL` and
  `STPSF_PATH`, and that `CRDS_PATH` is a per-user cache in the home directory (the
  `roman_notebooks` `refdata_dependencies.yaml` said "Nexus CRDS caches are local for now" in
  January 2026). Check with `env | grep -E 'CRDS|STPSF'`, including whether `CRDS_CONTEXT` is
  pinned.
- **They reach the `mejiro` kernel.** The variables are set for the default kernel. Confirm they
  also appear in a kernel made with `kernel-create`, both in a terminal after
  `source kernel-activate mejiro` and in a notebook (`os.environ`).
- **STPSF data version.** The pinned `stpsf==2.3.0` needs data 2.3.0 or later. Compare
  `cat $STPSF_PATH/version.txt` with `stpsf.DATA_VERSION_MIN`. If the Nexus's data is older,
  check that the page's two ways of pointing a kernel at other data (`kernel.json` `env`, or
  `os.environ` before importing stpsf) work.
- **Installing.** `kernel-create mejiro 3.13 "mejiro"`, then `pip install -e ".[dev,roman]"`,
  including the GitHub-hosted `slsim`, `speclite` and `skypy` forks.
- **Tests.** The CI `pytest` command and `pytest --run-crds tests/test_pipeline/test_romanisim_crds.py`,
  with how long each takes and on which server size.
- **CRDS cache size.** How much `exposure.usecrds: true` adds to `CRDS_PATH` per detector and
  band (the page says "several GB per detector"), and whether that's a problem for home
  directory storage.
- **The notebook.** `mejiro-projects/lens_search/lens_search_hlwas.ipynb` on a medium server
  with one skycell and one filter, including where the HLWAS manifests are on the Nexus (the
  notebook falls back to downloading them from GitHub) and the free space on the pod-local
  scratch directory it writes L2 copies to.

The notebook and `exposure.usecrds` also work around two romanisim 0.14 problems with CRDS; see
`romanisim_crds_frame_time.md`.
