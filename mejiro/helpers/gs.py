import galsim
import random
from galsim import roman
from galsim import InterpolatedImage, Image


def get_random_hlwas_wcs(detector, suppress_output=False):
    ra = random.uniform(15, 45)
    dec = random.uniform(-45, -15)

    # TODO fix
    ra_targ = galsim.Angle.from_hms('16:01:41.01257')
    dec_targ = galsim.Angle.from_dms('66:48:10.1312')
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)

    wcs_dict = roman.getWCS(world_pos=targ_pos, SCAs=detector)

    if not suppress_output:
        print(f'RA: {ra}, DEC: {dec}')

    return wcs_dict[detector]


def get_bandpass(band):
    band = band.upper()
    translate: {
        'F087': 'Z087',
        'F106': 'Y106',
        'F129': 'J129',
        'F158': 'H158',
        'F184': 'F184',
        'F149': 'W149'
    }
    return translate[band]


def get_random_detector(suppress_output=False):
    detector = str(random.randint(1, 18))
    if not suppress_output:
        print(f'Detector: {detector}')
    return detector


def get_random_detector_pos(suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    x, y = random.randrange(4, 4092), random.randrange(4, 4092)
    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return galsim.PositionD(x, y)


def convolve(interp, bandpass, detector, detector_position, num_pix, pupil_bin=1):
    galsim_psf = roman.getPSF(detector, 
                              SCA_pos=detector_position, 
                              bandpass=None, 
                              wavelength=bandpass, 
                              pupil_bin=pupil_bin)

    # https://galsim-developers.github.io/GalSim/_build/html/composite.html#galsim.Convolve
    convolved = galsim.Convolve(interp, galsim_psf)

    # draw interpolated image
    im = galsim.ImageF(num_pix, num_pix, scale=0.11)
    im.setOrigin(0, 0)

    return convolved.drawImage(im)


def get_sky_bkg(wcs, band, exposure_time, num_pix, seed=None):
    # was only one band provided as a string? or a list of bands?
    single_band = False
    if not isinstance(bands, list):
        single_band = True
        bands = [bands]

    # set rng
    if seed is None:
        rng = galsim.UniformDeviate()
    else:
        rng = galsim.UniformDeviate(seed)

    bkgs = []
    for band in bands:
        bandpass = get_bandpass(band)

        # build Image
        sky_image = galsim.ImageF(num_pix, num_pix, wcs=wcs)

        SCA_cent_pos = wcs.toWorld(sky_image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos, exptime=exposure_time)
        sky_level *= (1.0 + roman.stray_light_fraction)
        wcs.makeSkyImage(sky_image, sky_level)

        thermal_bkg = roman.thermal_backgrounds[bandpass] * exposure_time
        sky_image += thermal_bkg

        poisson_noise = galsim.PoissonNoise(rng)
        sky_image.addNoise(poisson_noise)

        bkgs.append(sky_image)

    if single_band:
        return bkgs[0]
    else:
        return bkgs