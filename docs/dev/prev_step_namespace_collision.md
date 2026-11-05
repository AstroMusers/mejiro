# Bug: `--prev-step` collides across pipeline steps in `mejiro_pipeline.py`

Status: fixed in the orchestrator (`Pipeline._tail_args()`). The "Related" L3 limitation at
the end is still open.
Last verified: 2026-11-04.

Found 2026-07-19 while collapsing the `05_romanisim_l3` step directories into
`05_romanisim`.

## Resolution

`Pipeline._tail_args()` in `mejiro/pipeline/mejiro_pipeline.py` gives `calculate_snrs` and
`_06_h5_export` a copy of the args with `prev_step` set to the exposure step that ran
(`05_galsim` or `05_romanisim`), and leaves the original `'03'` that step 04 uses alone.
`run()`, `run_script('snr')` and `run_script(6)` all use it. This is a variant of the
"distinct value for the later steps" option below, without new flags.

Tests covering it, in `tests/test_pipeline/test_mejiro_pipeline.py`:

- `test_tail_args_point_later_steps_at_the_exposure_step` checks `_tail_args()` for both
  engines, and that step 04's `prev_step` is unchanged.
- `test_pipeline_run` runs the galsim path end to end and asserts that step 06 found
  exposures, which addresses the "silently-empty export" concern below.

The rest of this note is the original analysis, kept for history. Its line numbers are from
2026-07-19.

## Summary

`Pipeline` (then called `MejiroPipeline` in this note) builds a single `argparse.Namespace` and passes that same object to
every step's `main()`. Three different steps read `args.prev_step`, but they mean
different things by it, and only one of them is what the orchestrator's argument was
declared for.

- `mejiro/pipeline/mejiro_pipeline.py:53` declares `--prev-step`,
  `choices=['02', '03']`, `default='03'`, with the comment
  "`_04_create_synthetic_images` flag: which step to read lens pickles from".
- `mejiro/pipeline/_04_create_synthetic_images.py:111` consumes it as intended
  (`PipelineHelper(args, args.prev_step, ...)`).
- `mejiro/pipeline/_06_h5_export.py:65` does
  `prev_script_name = getattr(args, 'prev_step', None) or PREV_SCRIPT_NAME` — so when
  driven through the orchestrator it sees `'03'` and never falls back to its own
  `PREV_SCRIPT_NAME`.
- `mejiro/pipeline/calculate_snrs.py:52` has the identical line and the identical
  problem.

Net effect: running the pipeline through `mejiro_pipeline.py` (`run()`, or
`run_script(6)` / `run_script('snr')`) makes the export and SNR steps look for exposures
in `<pipeline_dir>/03/` — the subhalo directory — instead of the exposure step's output
directory. Running the scripts standalone from the CLI is unaffected, because there
`--prev-step` is unset and the `or PREV_SCRIPT_NAME` fallback fires. All production runs
to date have used the standalone shell scripts (`rung_0.sh` etc.), which is why this has
not bitten yet.

## Why the test suite does not catch it

`tests/test_pipeline/test_mejiro_pipeline.py` runs against `test.yaml`, whose
`imaging.engine` is `galsim`, and it passes today. That means step 06 is tolerating a
wrong input directory rather than erroring — worth confirming during the fix, since a
silently-empty export is a worse failure mode than a crash and may deserve its own guard.

## Suggested fix

Give each consumer its own flag name rather than sharing one:

- Rename the `_04` flag to something step-specific (e.g. `--lens-step`) in both
  `_04_create_synthetic_images.py:476` and `mejiro_pipeline.py:53`, or
- keep `--prev-step` on `_04` and register a distinct `--export-prev-step` /
  `--snr-prev-step` in the orchestrator, defaulting to `None` so the
  `or PREV_SCRIPT_NAME` fallback in `_06_h5_export.py` and `calculate_snrs.py` works as
  designed.

The second option is smaller and leaves the standalone CLIs untouched. Either way, add a
regression test that drives `Pipeline.run()` and asserts step 06 resolved its input
to the exposure step's directory, not `03/`.

## Related

`mejiro_pipeline.py:54-56` builds `arg_list` from only `--config` and `--data_dir`, so
`--level`, `--dither-pattern` and `--max-systems` registered at lines 48-51 always take
their defaults — the orchestrator can never run `--level l3`. That may be intentional,
but it means the L3 path is only reachable via the standalone script.
