"""One-time fix: rename romanisim exposure cutouts saved with a `.npz.npy` double
extension back to plain `.npy`.

Before the cutout-naming fix in `romanisim_pipeline.py`, lightweight `.npz` SyntheticImage
inputs produced files named `Exposure_..._{band}.npz.npy` (the `.npz` survived and
`np.save` appended `.npy`). This script strips the stray `.npz` so every cutout is a plain
`Exposure_..._{band}.npy`. New runs already write the correct name, so this is a one-shot
correction for existing data.

Set DRY_RUN = True to preview without renaming.
"""
import os
from glob import glob

from tqdm.auto import tqdm

DRY_RUN = False

ROOT = '/nfsdata1/bwedig/mejiro/roman_data_challenge_rung_1/05_romanisim'
SUFFIX = '.npz.npy'

files = sorted(glob(os.path.join(ROOT, 'sca*', f'Exposure_*{SUFFIX}')))
print(f'Found {len(files)} {SUFFIX} cutouts under {ROOT}')

renamed = 0
skipped = 0
for src in tqdm(files):
    dst = src[:-len(SUFFIX)] + '.npy'
    if os.path.exists(dst):
        print(f'SKIP (target exists): {dst}')
        skipped += 1
        continue
    if DRY_RUN:
        renamed += 1
        continue
    os.rename(src, dst)
    renamed += 1

action = 'Would rename' if DRY_RUN else 'Renamed'
print(f'{action} {renamed} files; skipped {skipped} (target already existed).')
if DRY_RUN:
    print('DRY_RUN is True — no files were changed. Set DRY_RUN = False to apply.')
