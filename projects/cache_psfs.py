import os
from multiprocessing import Pool

from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils import roman_util

os.nice(19)


def _cache_single_psf(task):
    band, detector, center = task
    psf_id = STPSFEngine.get_psf_id(band, detector, center, 5, 91)
    STPSFEngine.cache_psf(psf_id, '/data/bwedig/mejiro/cached_psfs')


def main():
    dry_run = False

    bands = ['F106', 'F129', 'F158']
    centers = roman_util.divide_up_sca(56)

    tasks = [
        (band, detector, center)
        for band in bands
        for detector in range(1, 19)
        for center in centers
    ]

    if dry_run:
        for band, detector, center in tasks:
            print(f"Band: {band}, Detector: {detector}, Center: {center}")
    else:
        total_tasks = len(tasks)
        completed = 0
        with Pool(processes=16) as pool:
            for _ in pool.imap_unordered(_cache_single_psf, tasks):
                completed += 1
                if completed % 50 == 0 or completed == total_tasks:
                    percent = 100 * completed / total_tasks
                    print(f"Progress: {completed}/{total_tasks} ({percent:.1f}%)")

    if dry_run:
        print(f"Will create {len(tasks)} PSFs.")


if __name__ == '__main__':
    main()
