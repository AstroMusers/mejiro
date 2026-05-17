"""
Scan a directory for corrupted pickle files and optionally delete them.

Corrupted pickles (e.g. from an interrupted write) raise EOFError when loaded.
After deleting them, re-running _04_create_synthetic_images.py with the same
config will regenerate only the missing files thanks to its skip-existing logic.

Usage:
    python3 scan_corrupted_pickles.py <directory> [--pattern GLOB] [--delete]

Arguments:
    directory: Path to scan (searches recursively through subdirectories).
    --pattern: Glob pattern for pickle filenames (default: SyntheticImage_*.pkl).
    --delete:  Delete corrupted files instead of just reporting them.
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

from tqdm import tqdm


def _check_pickle(path):
    try:
        with open(path, 'rb') as f:
            pickle.load(f)
        return path, None
    except Exception as e:
        return path, e


def scan(directory, pattern, delete, workers):
    search = os.path.join(directory, '**', pattern)
    paths = sorted(glob(search, recursive=True))

    if not paths:
        print(f'No files matching {pattern!r} found under {directory}')
        return

    corrupted = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_check_pickle, p): p for p in paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Scanning'):
                path, err = future.result()
                if err is not None:
                    corrupted.append((path, err))
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    print(f'\nScanned {len(paths)} file(s). Found {len(corrupted)} corrupted.')

    if not corrupted:
        return

    for path, err in corrupted:
        print(f'  CORRUPTED: {path}  ({type(err).__name__}: {err})')

    if delete:
        for path, _ in corrupted:
            os.remove(path)
            print(f'  Deleted: {path}')
        print(f'Deleted {len(corrupted)} file(s).')
    else:
        print('\nRe-run with --delete to remove these files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scan for and optionally delete corrupted pickle files.')
    parser.add_argument('directory', help='Root directory to search.')
    parser.add_argument('--pattern', default='SyntheticImage_*.pkl',
                        help='Glob pattern for filenames (default: SyntheticImage_*.pkl).')
    parser.add_argument('--delete', action='store_true',
                        help='Delete corrupted files after reporting them.')
    parser.add_argument('--workers', type=int, default=36,
                        help='Number of parallel worker processes (default: 36).')
    args = parser.parse_args()
    scan(args.directory, args.pattern, args.delete, args.workers)
