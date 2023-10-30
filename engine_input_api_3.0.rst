Currently Supported Calculation Modes (v3.0)
===============================================

The following following JWST instrument/mode combinations are currently implemented,
working, and part of the nightly regression tests:

* ``miri:``
    - imaging
    - imaging_ts
    - mrs
    - mrs_ts                                                                                                                                                                                                                                                                                                                         
    - coronagraphy
    - lrsslit
    - lrsslitless
    - target_acq

* ``nircam:``
    - sw_imaging
    - lw_imaging
    - sw_ts
    - lw_ts
    - ssgrism
    - wfgrism
    - coronagraphy
    - target_acq

* ``nirspec:``
    - ifu
    - msa
    - fixed_slit
    - target_acq
    - ifu_ver
    - mos_ver
    - mos_conf

* ``niriss:``
    - imaging
    - soss
    - ami
    - wfss
    - target_acq

The following Roman instrument/mode combinations are now implemented with Cycle 9
reference data:

* ``wfi:``
    - imaging
    - spectroscopy

Overview of Inputs
==================

The engine input api is a dict.  At the top level of the dict, there are:

background:
    string for canned background location lookup, or array of background values (see
    below)

background_level:
    string for canned background level (optional; see below)

configuration:
    dict: camera/telescope configuration

scene:
    list: list of sources

strategy:
    dict: strategy configuration

calculation:
    dict: test interface

fake_exception:
    test interface

error, server_test:
    reserved by server


Details of Inputs for ETC Calculation
=====================================

NOTE - keep all of the following in sync with: pandeia/engine/helpers/schema

As of now, the engine requires the following information to perform a calculation:

scene: list (no default)
  This is a list of Sources. Each Source is described by a dict with the following keys:

    position: dict
      Source position parameters described by the following keys:

        x_offset: float (default 0.0)
            Detector plane X offset from FOV center in arcseconds. Positive to the right.
            Coordinates are set such that at the center of the center pixel is 0,0
        y_offset: float (default 0.0)
            Detector plane Y offset from FOV center in arcseconds. Positive is up.
            Coordinates are set such that at the center of the center pixel is 0,0
        orientation: float (default 0.0)
            Detector plane orientation in degrees. Positive is in direction of +X.
            (e.g. orientation=90 is UP, and orientation=-90 is DOWN)

    shape: dict
      Source shape parameters described by the following keys:

        geometry: string (default "point")
            Supported geometries are "point", "gaussian2d", "flat", "sersic",
            "sersic_scale", and "power". Required additional source shape parameters are
            contingent on this parameter:

                "point" requires no additional parameters.

                "gaussian2d", "flat", "sersic", and "sersic_scale" all require these
                parameters:
                    major: float (default 0.1)
                        Semi-major axis in arcseconds. For "flat" this sets the size, for
                        "gaussian2d" this sets the sigma, for "sersic" this sets the
                        effective radius (within which half the flux is concentrated) and
                        for "sersic_scale" this sets a scale length where I(r) = I(0)/e.
                    minor: float (default 0.1)
                        Semi-minor axis in arcseconds
                    norm_method: string (default 'integ_infinity')
                        Methods of surface brightness normalization to perform. Supported
                        methods are: 
                        * integ_infinity: Normalize to the total intensity of the source, 
                        integrated to infinity 
                        * surf_center: Normalize to the surface brightness at the center 
                        of the source 
                        * surf_scale: Normalize to the surface brightness at the scale 
                        radius (for gaussian2d, 1-sigma; for sersic, the effective 
                        radius; for sersic_scale, the e-folding scale length; NOT 
                        AVAILABLE FOR FLAT SOURCES)
                    surf_area_units: string (default 'arcsec^2')
                        Specifies what area the flux to be renormalized in
                        spectrum/normalization/norm_flux is over.
                        * arcsec^2: the flux is per square arcsecond.
                        * sr: the flux is per steradian.

                "sersic" and "sersic_scale" require one additional parameter:
                    sersic_index: float (default 1.0)
                        Power law index that sets the shape of a sersic profile.
                        sersic_index = 1.0 --> exponential
                        sersic_index = 0.5 --> gaussian
                        sersic_index = 4.0 --> de Vaucouleurs

                "power" has its own parameters, different from the other profiles:
                    power_index: float
                        Power law index that sets the shape of the profile
                    r_core: float
                        Radius of the flat circular central core to which the profile 
                        is normalized
                    norm_method: string. 
                        For "power", must be set to 'surf_center'
                    surf_area_units: string (default 'arcsec^2'). 
                        Same as other profiles.

    spectrum: dict
      Source spectral parameters described by the following keys:

        redshift: float (default 0.0)
            Redshift to apply to the continuum. Since lines are added with physical units
            for their strength, they are added to the spectrum after normalization and
            redshift.

        extinction: dict
          Defines how the spectrum is reddened by interstellar dust

            law: string
                Extinction law to use. Supported laws are
                    * ``mw_rv_31`` - WD01 Milky Way curve for an R_V value of 3.1
                                     (default)
                    * ``mw_rv_40`` - WD01 Milky Way curve for an R_V value of 4.0
                    * ``mw_rv_55`` - WD01 Milky Way curve for an R_V value of 5.5
                    * ``hd210121`` - WD01 Extinction curve for high-latitude molecular
                                     cloud hd210121 with C/H = b_C = 40 ppm in log-normal 
                                     size dists
                    * ``lmc_avg``  - WD01 Average extinction curve for the LMC with C/H =
                                     b_C = 20 ppm in log-normal size dists
                    * ``lmc_2``    - WD01 LMC extinction curve with C/H = b_C = 10 ppm in
                                     log-normal size dists (30 Dor region)
                    * ``smc_bar``  - WD01 Extinction curve in SMC bar with C/H = b_C = 0
                                     ppm in log-normal size dists
                    * ``chapman09`` - Chapman et al. (2009) mid-IR extinction curve
                                     derived from three molecular clouds: Ophiuchus, 
                                     Perseus, and Serpens
            value: float
                Level of extinction in units of unit
            unit: string
                Units of extinction.  Allowed values are ``nh`` for hydrogen column
                density (cm^-2) and "mag" for magnitudes of extinction in specified
                bandpass, ext_bandpass
            bandpass: string
                Bandpass to which extinction is normalized to if unit="mag".  Allowed
                values are v, j, h, and k.

        normalization: dict
          Defines how the spectrum is to be scaled.

            type: string
                Method of normalization to perform.  Supported methods are
                    * ``at_lambda`` - Specify norm_flux in fluxunit at a specfic
                      wavelength, norm_wave
                    * ``hst`` - Specify a bandpass in the form of an STSynphot "obsmode"
                      string
                      (https://stsynphot.readthedocs.io/en/latest/stsynphot/obsmode.html)
                      to pass along to STSynphot along with fluxunit and norm_flux.  The
                      general form is "<instrument>,<detector>,<filter>". The Web UI lists
                      the most commonly-used options.
                    * ``jwst`` - Specify a bandpass as an instrument configuration in the
                      form of a comma-separated string <instrument>,<mode>,<filter> along
                      with fluxunit and norm_flux. Because these options read JWST data,
                      this option will not work with Roman unless the JWST data is also
                      present. The Web UI lists the most commonly-used options. 
                    * ``photsys`` - Specify bandpass in the form of a comma-separated
                      string <photsys>,<filter>
                      Options are:
                        * Bessell
                            - bessell,j
                            - bessell,h
                            - bessell,k
                        * 2MASS
                            - 2mass,j
                            - 2mass,h
                            - 2mass,ks
                        * WISE
                            - wise,w1
                            - wise,w2
                            - wise,w3
                            - wise,w4
                        * Gaia
                            - gaia,g
                        * GALEX
                            - galex,fuv
                            - galex,nuv
                        * Cousins
                            - cousins,r
                            - cousins,i
                        * Johnson
                            - johnson,u
                            - johnson,b
                            - johnson,v
                            - johnson,r
                            - johnson,i
                            - johnson,j
                            - johnson,h
                            - johnson,k
                        * MSX
                            - msx,a
                            - msx,b1
                            - msx,b2
                            - msx,c
                            - msx,d
                            - msx,e
                        * SDSS
                            - sdss,u
                            - sdss,g
                            - sdss,r
                            - sdss,i
                            - sdss,z
                        * Spitzer
                            - irac3.6
                            - irac4.5
                            - irac5.8
                            - irac8.0
                            - mips24

                    * ``none`` - Do not normalize spectrum.  Only valid for a spectrum
                      type of 'input'.

            norm_wave: float
                Reference wavelength in 'norm_waveunit' at which spectrum will be scaled
                for type 'at_lambda'. Ignored for other normalization types.
            norm_waveunit: string
                Specify the wavelength units used in normalization for type 'at_lambda'
            norm_flux: float
                Reference flux in 'norm_fluxunit' to which spectrum will be scaled.
            norm_fluxunit: string
                Specify the flux units in which the normalization should occur.
                Supports flam, fnu, vegamag, abmag, mjy, ujy, njy, jy
            bandpass: string
                Specifies the key used to obtain the normalization bandpass for
                types 'hst', 'jwst', and 'photsys'.

        sed: dict
          Defines the spectral energy distribution of the spectrum.

            sed_type: string
                Type of the spectral energy distribution. Each type requires its own set
                of parameters. The analytic sed_types (none, flat, powerlaw, flat) all
                require 'wmin', 'wmax', and 'sampling' to define the range and wavelength
                sampling over which the model spectrum is calculated. However, these
                parameters are only available in the API for testing purposes and cannot
                be configured via the Web UI.

                Analytic:

                    **no_continuum** - No continuum, specifically Flux = 0.0 over
                    specified range [wmin, wmax]
                        wmin: float (default 0.02)
                            Minimum wavelength in microns
                        wmax: float (default 35.0)
                            Maximum wavelength in microns
                        sampling: int (default 200)
                            Sets the logarithmic wavelength sampling of the model spectrum

                    **flat** - Flat spectrum in specified units calculated over specified
                    range [wmin, wmax]
                        wmin: float (default 0.02)
                            Minimum wavelength in microns
                        wmax: float (default 35.0)
                            Maximum wavelength in microns
                        sampling: int (default 200)
                            Sets the logarithmic wavelength sampling of the model spectrum
                        unit: string
                            Units of spectrum, either 'fnu' or 'flam'

                    **powerlaw** - Powerlaw spectrum where F ~ lambda ^ index calculated
                    over range [wmin, wmax]
                        wmin: float (default 0.02)
                            Minimum wavelength in microns
                        wmax: float (default 35.0)
                            Maximum wavelength in microns
                        sampling: int (default 200)
                            Sets the logarithmic wavelength sampling of the model spectrum
                        unit: string
                            Units of spectrum, either 'fnu' or 'flam'
                        index: float
                            Exponent of the power law

                    **blackbody** - Blackbody spectrum calculated over range [wmin, wmax]
                        wmin: float (default 0.02)
                            Minimum wavelength in microns
                        wmax: float (default 35.0)
                            Maximum wavelength in microns
                        sampling: int (default 200)
                            Sets the logarithmic wavelength sampling of the model spectrum
                        temp: float
                            Temperature of the blackbody in Kelvin

                Grid-type:

                    **phoenix** - Parameterized stellar atmosphere models calculated by
                    the Phoenix group
                        key: string
                            In webapp mode, a key is used to look up a predefined set of
                            parameters. If not in webapp mode and if key is not provided,
                            model parameters can be passed directly:
                        teff: float
                            Effective temperature. Allowed range is 2000 K to 70000 K
                        log_g: float
                            Surface gravity in log10(cgs) units. Allowed range is 0.0 to 5.5.
                        metallicity: float
                            Metallicity in units of log10(solar metallicity). Allowed
                            range is -4.0 to +0.5

                    **k93models** - Parameterized stellar atmosphere models from Kurucz
                    and Castelli 1993
                        key: string
                            In webapp mode, a key is used to look up a predefined set of
                            parameters. If not in webapp mode and if key is not provided,
                            model parameters can be passed directly:
                        teff: float
                            Effective temperature. Allowed range is 2000 K to 70000 K
                        log_g: float
                            Surface gravity in log10(cgs) units. Allowed range is 0.0 to 5.5.

                    **ck04models** - Parameterized stellar atmosphere models from Castelli & 
                    Kurucz 2004
                        key: string
                            In webapp mode, a key is used to look up a predefined set of
                            parameters. If not in webapp mode and if key is not provided,
                            model parameters can be passed directly:
                        teff: float
                            Effective temperature. Allowed range is 2000 K to 70000 K
                        log_g: float
                            Surface gravity in log10(cgs) units. Allowed range is 0.0 to 5.5.


                Library-type:

                    **brown2014** - Integrated spectra of galaxies from Brown et al. (2014)
                        key: string
                            Key used to look up which spectrum to load.

                    **brown2019** - Integrated spectra of galaxies from Brown et al. (2019)
                        key: string
                            Key used to look up which spectrum to load.

                    **bt_settl** - Cool dwarf models from BT-Settl (Allard+ 2015)
                        key: string
                            Key used to look up which spectrum to load.

                    **bz77** - The Bruzual Atlas of 77 stellar spectra for galaxy spectral 
                    synthesis
                        key: string
                            Key used to look up which spectrum to load.

                    **cool_dwarfs** - Low-temperature ATMO2020 spectral models
                        key: string
                            Key used to look up which spectrum to load.

                    **hst_calspec** - HST standard star spectra
                        key: string
                            Key used to look up which spectrum to load.

                    **pickles** - Stellar Spectral Flux Library by A.J. Pickles
                        key: string
                            Key used to look up which spectrum to load.

                    **pne** - Planetary Nebula spectra from CLOUDY
                        key: string
                            Key used to look up which spectrum to load.

                    **nonstellar** - Assorted brown dwarf, nebulae, star formation regions, 
                    galaxies. From the original HST ETC.
                        key: string
                            Key used to look up which spectrum to load.

                    **novae** - Stellar Novae spectra
                        key: string
                            Key used to look up which spectrum to load.
                    
                    **qso** - Composite QSO spectrum
                        key: string
                            Key ("qso") used to look up the spectrum.

                    **stellar_pop** - Simple Stellar Populations based on the FSPS models
                    (Conroy et al. 2009), MIST isochrones, BaSeL library, Kroupa IMF,
                    CLOUDY nebular emission, and AGB circumstellar dust emission from
                    Villaume et al. (2014)
                        key: string
                            Key used to look up which spectrum to load.

                    **sun_planets** - Spectra of the Sun and giant planets
                        key: string
                            Key used to look up which spectrum to load.

                    **swire** - Normal galaxy spectra from the SWIRE template library
                        key: string
                            Key used to look up which spectrum to load.

                Input:

                    **input** - spectrum provided via input arrays
                        spectrum: list-like or numpy.ndarray
                            The 0th index is taken to be wavelength in units of 'microns'.
                            The 1st index is taken to be the flux in units of 'mJy'.

        lines: list (default [])
          List of line definitions. Each definition is a dict with keys:

              name: string (default 'no name')
                  Name of line (e.g. 'Hydrogen Alpha')
              center: float (default 5.2)
                  Wavelength at line center in w_unit
              strength: float (default 1.0e-14)
                  Strength of line in erg/cm^2/s for emission or
                  optical depth for absorption
              profile: string
                  Line profile type:
                    * gaussian      *default*
                    * voigt          NOT YET IMPLEMENTED
              emission_or_absorption: string
                  Line type:
                    * emission      *default* (mJy)
                    * absorption    (tau)

            A profile type of **gaussian** (currently the only type) requires one
            additional parameter:

              width: float (default 200.0)
                  Full-width half-max of line in km/s

            When implemented, profile type of **voigt** will require two additional
            parameters:

              gaussian_fwhm: float (default 200.0)
                  Full-width half-max of the gaussian core of the line in units of km/s
              lorentzian_fwhm: float (default 500.0)
                  Full-width half-max of the lorentzian wings of the line in units of km/s

background: string (default 'minzodi') or list-like or numpy.ndarray
  Possible string values are: none, minzodi, and ecliptic.  String values trigger the use
  of a canned background model at the location given. If a background spectrum is
  provided, it is assumed that the 0th index is the wavelength in microns and the 1st
  index is the background surface brightness in MJy/sr.

background_level: string (default 'benchmark').
  Possible string values are "high", "medium", "low", and (only for minzodi location)
  "benchmark". This value is only used if background is a string that's not none.

calculation: dict
  Set of parameters to toggle the inclusion of different effects and noise parameters in a
  calculation. This section is optional and largely for testing purposes. These are not
  supported in the Web UI. If the parameter is None, the default from the instrument
  configuration is used. If the parameter is set to True or False the effect is switched
  on or off, overriding the instrument default configurations. Use at your own risk.

    noise: dict
      Noise components

        crs: bool/None
            Cosmic rays
        dark: bool/None
            Detector Dark Current
        excess: bool/None
            Detector excess noise parameters
        ffnoise: bool/None
            Flat-field noise
        readnoise: bool/None
            Detector Read noise
        scatter: bool/None
            Echelle Scattering noise

    effects: dict
      Effects that can affect the noise or detector response or both

        saturation: bool/None
            Pixel saturation


configuration: dict
  This is the configuration for the instrument and detector, using the following keys:

    instrument: dict
      The instrument configuration parameters

        instrument: string
          for JWST:
            * miri
            * nircam
            * nirspec
            * niriss

          for Roman:
            * wfi

        mode: string
          valid modes:
          for JWST:
            * imaging
            * imaging_ts
            * sw_imaging
            * lw_imaging
            * msa (called mos in the webapp)
            * mos_ver
            * mos_conf
            * mrs
            * mrs_ts
            * soss
            * ifu
            * ifu_ver
            * wfss
            * ssgrism (called lw_tsgrism in the webapp)
            * sw_ts
            * lw_ts
            * wfgrism
            * lrsslit
            * lrsslitless
            * fixed_slit
            * bots
            * ami
            * coronagraphy
            * target_acq

          for Roman:
            * imaging
            * spectroscopy

        filter: string
           (e.g. f070w)

        disperser: string
           (e.g. g235h)

        aperture: string
           (e.g. a200s1)

        detector: string (only valid for NIRCam Coronagraphic Imaging) 
            Identifies which detector (e.g. sw) coronagraphic imaging should take place
            on, and therefore whether it is primary or secondary coronagraphy.

        shutter_location: string (only valid for NIRSpec MSA, MOS_CONF, and MOS_VER modes)
            Identifier string for slitlet position to use for MSA calculation

        slitlet_shape: string or list (only valid for NIRSpec MSA, MOS_CONF, and MOS_VER
        modes)
            A string denoting a slitlet grid shape. Can also be a list of 2-element
            offsets describing set of shutters to be open. Offsets are from scene center
            in units of shutter spacing.
                (e.g. slitlet_shape = [[0,-2],[0,0],[0,2]])

    detector: dict
      Exposure configuration parameters.

        subarray: string
           full, 64x64, etc.; Instrument-dependent
        readout_pattern: string
           Instrument-dependent
        ngroup: int
           Number of groups
        nint: int
           Number of integrations
        nexp: int
           Number of exposures

    dynamic_scene: boolean
        Toggle whether to allow the size of the scene to expand dynamically to include all
        configured sources.

    scene_size: float
        Default size of the scene in arcseconds. Used if dynamic_scene is True.

    max_scene_size: float
        Maximum allowable scene_size in arcseconds.

strategy: dict
  Configuration parameters for observing strategy.

    method: string
        Instrument and mode dependent. Currently supported methods are:
            * imagingapphot
            * specapphot
            * coronagraphy
            * ifuapphot
            * ifunodinscene
            * ifunodoffscene
            * msafullapphot
            * msaapphot
            * msashutterapphot
            * soss
            * taphot
            * tacentroid

        Planned methods that are not yet implemented include:
            imagingoptphot, specoptphot, speclinephot
        In most cases, only one extraction strategy is valid for a given mode.

    units: string  (default: "arcsec")
        Angular units used by the strategy (for aperture_size and sky_annulus). Valid
        option is "arcsec".
    target_source: string
        Sent by the Web UI, but currently unused by the engine
    target_type: string
        Sent by the Web UI, but currently unused by the engine

    The rest of the parameters will be method dependent.  
    
    The parameters required for **imagingapphot**, **msaapphot**, and **ifuapphot** are:

        background_subtraction: boolean
            Toggle whether sky annulus background subtraction (True) or ideal noiseless
            background subtraction (False) is performed.
        aperture_size: float
            Radius of extraction aperture in "units". 
        sky_annulus: list-like of format (float, float)
            The inner and outer radii in "units" of the sky region used for background
            subtraction. Only used if background_subtraction=True.
        target_xy: list-like of format (float, float)
            X and Y center position of the aperture and sky annulus, in arcseconds.

    An additional parameter required for **imagingapphot** only is:
        is_aperture_ee: boolean
            Must be set to False.

    The common parameter required for all spectroscopic modes (**specapphot**,
    **msafullapphot**, **soss**, **ifuapphot**, **ifunodinscene**, **ifunodoffscene**) is:

        reference_wavelength: float
            Wavelength in microns at which the scalar parameters should be extracted from.
            The values will be extracted from the pixel closest to this requested value.

    The parameters required for **specapphot** are:
        background_subtraction: boolean
            Toggle whether sky annulus background subtraction (True) or ideal noiseless
            background subtraction (False) is performed.
        aperture_size: float
            Size of extraction aperture in "units". Defines a rectangular extraction
            region that is the aperture_size in height, and the width of the trace in
            length. Note that as of v2.0 this refers to the FULL height of a spectroscopic
            aperture, not a half height. 
        sky_annulus: list-like of format (float, float)
            Size of the sky region in "units." Defines two rectangular extraction regions
            that extend the width of the trace in length, and from (first value) to
            (second value) in height, mirrored across the centerline of the trace. Only
            used if background_subtraction=True.
        target_xy: list-like of format (float, float)
            X and Y center position of the aperture and sky annulus, in arcseconds.

    The parameters required for **ifunodinscene** and **ifunodoffscene** are:

        aperture_size: float
            Radius of extraction aperture(s) in "units"
        reference_wavelength: float
            Wavelength in microns at which the scalar parameters should be extracted from.
            The values will be extracted from the pixel closest to this requested value. 
        target_xy: list-like of format (float, float)
            X and Y center position of the aperture and sky annulus, in arcseconds.
        dithers: list of dicts with format {'x': <float>, 'y': <float>}
            Dither positions given in "units" from center of the Scene. This will be used
            to define the location of the second aperture that the source will be moved to
            for the second dither (IFUNodInScene) or the amount by which the source will be
            offset to use the single aperture for a sky measurement (IFUNodOffScene)

    The parameters required for **msafullapphot**, **msaapphot**, and **msashutterapphot**
    are:

        shutter_offset: list-like of format (float, float)
            Offset of shutter pattern from center of scene in "units"
        dithers: list of dicts
            Dither positions and MSA shutter configuration with the following format:
                x: float
                    X position of the central shutter (the target position), in
                    arcseconds.
                y: float
                    Y position of the central shutter (the target position), in
                    arcseconds.
                on_source: list of bool
                    List of booleans denoting whether a shutter should be treated as
                    source or sky. Must specify the same number of shutter positions as
                    the slitlet_shape. (not necessary for MSAApPhot)

    The parameters required for **soss** are:
        order: int
            Specify which order to extract. Can be 1 or 2.

    The parameters required for **coronagraphy** are:

        target_xy: two-element list-like (float, float)
            Position of extraction aperture
        aperture_size: float
            Radius of extraction aperture in 'units'
        sky_annulus: two-element list-like (float, float)
            Inner and outer radii of sky background estimation region in 'units'
        contrast_azimuth: float
            Azimuth (west of north) at which to calculate contrast curve
        calc_type: string
            Set to "contrast", identifies that this mode requires a contrast calculation.
        pointing_error: two-element list-like (float, float)
            Amount to shift occulted source to emulate imperfect pointing
        delta_opd: float
            Change in system OPD
        scene_rotation: float, degrees
            Rotation angle to apply to scene
        psf_subtraction: string
            Can be set to "optimal" (for autoscaling subtraction), "no_autoscale" (for
            subtraction without autoscaling), "target_only" (for the science scene only
            and no subtraction), or "psf_only" (for the PSF subtraction source only, and
            no subtraction)
        psf_subtraction_source: Complete source dict in engine API format (see above)
            Definition of source to use for PSF subtraction. This must be set here rather
            than as a source in the scene. Position parameters must be specified, though
            they are ignored. Use psf_subtraction_xy to specify the location of the psf
            subtraction source relative to the scene.
        psf_subtraction_xy: two-element list-like (float, float)
            Offset to apply to psf_subtraction_source, in 'units'
        unocculted_xy: two-element list-like (float, float)
            Offset to apply to source to measure contrast between occulted and unocculted
            observation, in 'units'

    The parameters required for **taphot** are:

        target_xy: list-like of format (float, float)
            X and Y center position of the aperture and sky annulus, in arcseconds.


    The parameters required for **tacentroid** are:

        target_xy: list-like of format (float, float)
            X and Y center position of the aperture and sky annulus, in arcseconds.
        axis: string
            Direction the centroid is calculated, "x" or "y"

fake_exception: list of strings
    If present, this list is searched for control terms that cause perform_calculation to
    raise exceptions for testing purposes. Currently recognized strings are:

        'pandeia':
             raise PandeiaException

        'exception':
             raise Exception

    Other strings may be added later to add exceptions or modify the details of the
    exception objects raised.
