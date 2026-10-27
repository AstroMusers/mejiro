# `--prev-step` collision across pipeline steps in `mejiro_pipeline.py`

The namespace collision was fixed in `b6b9d86c`. The orchestrator now gives export
and SNR steps a copied argument namespace whose `prev_step` names the configured
engine's exposure directory. Step 04 retains its original lens-pickle input.

Eleven focused dispatch tests cover both engines, both lens-input steps, full and
individual-step dispatch, namespace isolation, and rejection of unsupported engines
in the exposure/export tail. Their step entry points are mocked. They exercise the
actual orchestrator without running detector simulations or validating HDF5 output.

The detector integration test cannot currently collect in the shared environment
because `stpsf` is unavailable. Its HDF5-content assertions still need a run with
the complete detector dependencies and input data.

## Historical behavior

The collision was found on 2026-07-19 while collapsing the `05_romanisim_l3`
step directories into `05_romanisim`. The following describes the original failure.

`MejiroPipeline` builds a single `argparse.Namespace` and passes that same object to
every step's `main()`. Three different steps read `args.prev_step`, but they mean
different things by it, and only one of them is what the orchestrator's argument was
declared for.

- `mejiro/pipeline/mejiro_pipeline.py:53` declares `--prev-step`,
  `choices=['02', '03']`, `default='03'`, with the comment
  "`_04_create_synthetic_images` flag: which step to read lens pickles from".
- `mejiro/pipeline/_04_create_synthetic_images.py:111` consumes it as intended
  (`PipelineHelper(args, args.prev_step, ...)`).
- `mejiro/pipeline/_06_h5_export.py:65` does
  `prev_script_name = getattr(args, 'prev_step', None) or PREV_SCRIPT_NAME` – so when
  driven through the orchestrator it sees `'03'` and never falls back to its own
  `PREV_SCRIPT_NAME`.
- `mejiro/pipeline/calculate_snrs.py:52` has the identical line and the identical
  problem.

Net effect: running the pipeline through `mejiro_pipeline.py` (`run()`, or
`run_script(6)` / `run_script('snr')`) makes the export and SNR steps look for exposures
in `<pipeline_dir>/03/` – the subhalo directory – instead of the exposure step's output
directory. Running the scripts standalone from the CLI is unaffected, because there
`--prev-step` is unset and the `or PREV_SCRIPT_NAME` fallback fires. All production runs
to date have used the standalone shell scripts (`rung_0.sh` etc.), which is why this has
not bitten yet.

## Original coverage gap

`tests/test_pipeline/test_mejiro_pipeline.py` uses the GalSim engine. It now checks
that the exported HDF5 image group is nonempty and that each exposure has the
expected units and exposure time. The focused dispatch tests additionally check
the input namespace before a downstream step runs. They do not replace the
detector integration test.

## Resolution

`Pipeline._tail_args()` copies `self.args` and selects the exposure producer's
`SCRIPT_NAME`, either `05_galsim` or `05_romanisim`. The full pipeline passes this
copy to export and, for Romanisim, to the SNR calculation. `run_script(6)` and
`run_script('snr')` use the same helper.

This keeps the standalone CLIs and step 04's `--prev-step` behavior unchanged.
Changing the copied namespace does not change the original lens-input namespace.

## Related

`mejiro_pipeline.py:54-56` builds `arg_list` from only `--config` and `--data_dir`, so
`--level`, `--dither-pattern` and `--max-systems` registered at lines 48-51 always take
their defaults – the orchestrator can never run `--level l3`. That may be intentional,
but it means the L3 path is only reachable via the standalone script.
