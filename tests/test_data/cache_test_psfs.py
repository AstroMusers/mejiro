from mejiro.engines.stpsf_engine import STPSFEngine


def main():
    bands = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']

    for band in bands:
        psf_id = STPSFEngine.get_psf_id(band, 1, (2044, 2044), 1, 41)
        STPSFEngine.cache_psf(psf_id, '')


if __name__ == '__main__':
    main()
