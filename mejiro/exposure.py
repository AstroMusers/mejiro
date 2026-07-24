import json
import logging
import os
import numpy as np
import time
import warnings

from mejiro.utils import util
from mejiro.analysis.snr_calculation import get_snr

logger = logging.getLogger(__name__)

EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION = 1


class Exposure:

    def __init__(self,
                 synthetic_image,
                 exposure_time,
                 engine='galsim',
                 engine_params={}
                 ):
        """
        Parameters
        ----------
        synthetic_image : SyntheticImage
            Source image in counts/sec.
        exposure_time : float
            Exposure time in seconds.
        engine : str
            Detector-effects engine. Determines the units of ``self.data``:

            * ``'galsim'``: **Counts** (= DN for Roman, where gain = 1.0 e-/DN).
              Computed as ``synthetic_image.data * exposure_time`` with sky, Poisson,
              dark, and read noise added, then divided by instrument gain.
            * ``'romanisim'``: **DN/s** (calibrated level-2 rate image, romanisim
              gain = 2 e-/DN). Divide by exposure time and multiply by 2 to convert
              to the galsim-engine electron scale.
        engine_params : dict, optional
            Engine-specific configuration.
        """
        start = time.time()

        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.engine = engine
        self.noise = None

        if engine not in synthetic_image.instrument.engines:
            raise ValueError(f"Engine '{engine}' is not supported by {synthetic_image.instrument.name}. Supported engines: {synthetic_image.instrument.engines}")

        if engine == 'galsim':
            from mejiro.engines.galsim_engine import GalSimEngine

            self.noise = GalSimEngine.get_empty_image(self.synthetic_image.num_pix,
                                                       self.synthetic_image.pixel_scale)

            if self.synthetic_image.instrument_name == 'Roman':
                # get exposure
                results, self.sky_background, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = GalSimEngine.get_roman_exposure(
                    synthetic_image, exposure_time, engine_params)

                # sum noise
                if self.sky_background is not None: self.noise += self.sky_background
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.reciprocity_failure is not None: self.noise += self.reciprocity_failure
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.nonlinearity is not None: self.noise += self.nonlinearity
                if self.ipc is not None: self.noise += self.ipc
                if self.read_noise is not None: self.noise += self.read_noise

            elif self.synthetic_image.instrument_name == 'HWO' or self.synthetic_image.instrument_name == 'JWST' or self.synthetic_image.instrument_name == 'HST':
                # get exposure
                results, self.sky_background, self.poisson_noise, self.dark_noise, self.read_noise = GalSimEngine.get_exposure(
                    synthetic_image, exposure_time, engine_params)
                
                # sum noise
                if self.sky_background is not None: self.noise += self.sky_background
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.read_noise is not None: self.noise += self.read_noise
                
            else:
                self.instrument_not_available_error(engine)

            # write the noise out to a numpy array
            self.noise = self.noise.array  # it's confusing for all detector effects to be type galsim.Image and the noise attribute to be an ndarray, but for comparison across engines, the noise should be an array and the detector effects should be Images so they can be passed in as engine params

        elif engine == 'lenstronomy':
            raise NotImplementedError('Lenstronomy engine not yet implemented')

            from mejiro.engines.lenstronomy_engine import LenstronomyEngine

            self.noise = np.zeros_like(self.synthetic_image.data)

            # get exposure
            results, self.noise = LenstronomyEngine.get_exposure(
                synthetic_image=synthetic_image,
                exposure_time=exposure_time,
                engine_params=engine_params)
            # TODO conditional for supported instruments

        elif engine == 'pandeia':
            raise NotImplementedError('Pandeia engine not yet implemented')
        
            from mejiro.engines.pandeia_engine import PandeiaEngine

            # get exposure
            results, self.noise = PandeiaEngine.get_roman_exposure(synthetic_image, exposure_time, engine_params)

            # TODO temporarily set noise to zeros until I can grab the noise that Pandeia generates
            self.noise = np.zeros((self.synthetic_image.num_pix, self.synthetic_image.num_pix))

        elif engine == 'romanisim':
            raise NotImplementedError('romanisim engine not yet implemented')
            
            from mejiro.engines.romanisim_engine import RomanISimEngine

            if self.synthetic_image.instrument_name == 'Roman':
                results, self.noise = RomanISimEngine.get_roman_exposure(synthetic_image, exposure_time, engine_params)
                
            else:
                self.instrument_not_available_error(engine)

        else:
            raise ValueError(f'Engine "{engine}" not recognized')

        # once engine params have been defaulted and validated, set them as an attribute
        self.engine_params = engine_params

        # set image and expoure attributes
        if self.engine == 'galsim':
            if self.synthetic_image.pieces:
                self.image, self.lens_image, self.source_image = results
                self.data, self.lens_data, self.source_data = self.image.array, self.lens_image.array, self.source_image.array
            else:
                self.image, self.lens_image, self.source_image = results, None, None
                self.data, self.lens_data, self.source_data = self.image.array, None, None
        else:
            if self.synthetic_image.pieces:
                self.data, self.lens_data, self.source_data = results
            else:
                self.data, self.lens_data, self.source_data = results, None, None

        # crop off edge effects (e.g., IPC)
        # TODO crop_edge_effects returns a new (cropped) array but the result is
        # discarded here, so self.data is left at its original size. Fixing this
        # requires also cropping self.noise / self.image / self.lens_image /
        # self.source_image to keep shapes consistent across the Exposure object.
        Exposure.crop_edge_effects(self.data, pad=3)

        # check for negative pixels
        if np.any(self.data < 0):
            warnings.warn(f'Negative pixel values in final image. Setting {np.sum(self.data < 0)} pixels to 0')
            self.data[self.data < 0] = 0

        if self.synthetic_image.pieces:
            # TODO same discarded-return issue as above
            Exposure.crop_edge_effects(self.lens_data, pad=3)
            Exposure.crop_edge_effects(self.source_data, pad=3)
            if np.any(self.lens_data < 0):
                warnings.warn(f'Negative pixel values in lens image. Setting {np.sum(self.lens_data < 0)} pixels to 0')
                self.lens_data[self.lens_data < 0] = 0
            if np.any(self.source_data < 0):
                warnings.warn(f'Negative pixel values in source image. Setting {np.sum(self.source_data < 0)} pixels to 0')
                self.source_data[self.source_data < 0] = 0

        end = time.time()
        self.calc_time = end - start
        logger.info(f'Exposure calculation time with {self.engine} engine: {util.calculate_execution_time(start, end, unit="s")}')

    def __getstate__(self):
        state = self.__dict__.copy()
        # drop the SyntheticImage reference on pickle: it's already written to disk in step 04,
        # so embedding it here fully duplicates that output (~2.77 MB per Exposure).
        # step 06 loads the SyntheticImage from step 04's pickle directly.
        state['synthetic_image'] = None
        return state

    def save_lightweight(self, path):
        """Write the compact ``.npz`` representation of this galsim exposure.

        Stores the final ``data`` array -- and, when the exposure was built with
        ``pieces=True``, the isolated ``lens_data`` / ``source_data`` arrays -- as
        ``float32`` plus a small JSON metadata blob carrying the scalars
        downstream consumers read (``_06_h5_export`` reads ``data``; the
        ``view_05`` viewer and :func:`mejiro.analysis.snr_calculation.get_snr`
        read the pieces, exposure time, band, etc.). The heavy galsim ``Image``
        objects and the summed ``noise`` array are intentionally dropped --
        nothing reads them back from disk. Loaders should use
        :func:`mejiro.utils.util.load_exposure`, which returns a
        :class:`LightweightExposure` for ``.npz`` paths.

        Parameters
        ----------
        path : str
            Destination path. Should end in ``.npz``.

        Notes
        -----
        Requires a live ``self.synthetic_image`` (present in ``_05_galsim`` right
        after construction, before the object would ever be pickled and its
        ``synthetic_image`` nulled by :meth:`__getstate__`). The band, instrument,
        and lens scalars are read from it.
        """
        si = self.synthetic_image
        sl = si.strong_lens

        pieces = self.lens_data is not None
        meta = {
            'schema_version': EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION,
            'band': str(si.band),
            'instrument_name': str(si.instrument_name),
            'num_pix': int(si.num_pix),
            'pixel_scale': float(si.pixel_scale),
            'exposure_time': float(self.exposure_time),
            'engine': str(self.engine),
            'pieces': bool(pieces),
            'lens': {
                'name': str(sl.name),
                'z_lens': float(sl.z_lens),
                'z_source': float(sl.z_source),
            },
        }
        meta_bytes = json.dumps(meta).encode('utf-8')

        arrays = {
            'data': np.ascontiguousarray(self.data, dtype=np.float32),
            'meta': np.frombuffer(meta_bytes, dtype=np.uint8),
        }
        if pieces:
            arrays['lens_data'] = np.ascontiguousarray(self.lens_data, dtype=np.float32)
            arrays['source_data'] = np.ascontiguousarray(self.source_data, dtype=np.float32)

        # Write atomically: np.savez auto-appends ".npz" when given a path string,
        # which makes atomic rename awkward, so pass an open file handle instead.
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)

    def get_snr(self, snr_per_pixel_threshold=1):
        return get_snr(self, snr_per_pixel_threshold=snr_per_pixel_threshold)[0]

    def plot(self, show_snr=False, savepath=None):
        import matplotlib.pyplot as plt

        plt.imshow(np.log10(self.data), origin='lower')

        title = f'{self.synthetic_image.strong_lens.name} (' + r'$z_{l}=$' + f'{self.synthetic_image.strong_lens.z_lens:.2f}, ' + r'$z_{s}=$' + f'{self.synthetic_image.strong_lens.z_source:.2f}' + f')\n{self.synthetic_image.instrument_name} {self.synthetic_image.band}, {self.exposure_time} s'
        if show_snr:
            snr = self.get_snr()
            title += f'\nSNR: {snr:.2f}'
        plt.title(title)
        cbar = plt.colorbar()
        cbar.set_label(r'log$_{10}$(Counts)')
        plt.xlabel('x [Pixels]')
        plt.ylabel('y [Pixels]')
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

    # def detailed_plot(self, savepath=None):
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #     axs[0].imshow(np.log10(self.data), cmap='viridis')
    #     axs[0].set_title(f'Exposure: {self.synthetic_image.instrument_name} {self.synthetic_image.band} band, {self.exposure_time} s')
    #     axs[0].set_xlabel('x [Pixels]')
    #     axs[0].set_ylabel('y [Pixels]')
    #     cbar = fig.colorbar(axs[0].images[0], ax=axs[0])
    #     cbar.set_label(r'log$_{10}$(Counts)')

    #     if self.lens_data is not None:
    #         axs[1].imshow(np.log10(self.lens_data), cmap='viridis')
    #         axs[1].set_title('Lens Image')
    #         axs[1].set_xlabel('x [Pixels]')
    #         axs[1].set_ylabel('y [Pixels]')
    #         cbar = fig.colorbar(axs[1].images[0], ax=axs[1])
    #         cbar.set_label(r'log$_{10}$(Counts)')

    #     if self.source_data is not None:
    #         axs[2].imshow(np.log10(self.source_data), cmap='viridis')
    #         axs[2].set_title('Source Image')
    #         axs[2].set_xlabel('x [Pixels]')
    #         axs[2].set_ylabel('y [Pixels]')
    #         cbar = fig.colorbar(axs[2].images[0], ax=axs[2])
    #         cbar.set_label(r'log$_{10}$(Counts)')

    #     plt.tight_layout()
    #     if savepath is not None:
    #         plt.savefig(savepath)
    #     plt.show()

    def instrument_not_available_error(self, engine):
        raise ValueError(
            f'Instrument "{self.synthetic_image.instrument_name}" not available for engine "{engine}."')

    @staticmethod
    def crop_edge_effects(image, pad):
        num_pix = image.shape[0]
        assert num_pix % 2 != 0, 'Image has even number of pixels'
        output_num_pix = num_pix - 2 * pad
        return util.center_crop_image(image, (output_num_pix, output_num_pix))


class LightweightExposure:
    """In-memory shim returned when loading a lightweight galsim exposure ``.npz``.

    Quacks like :class:`Exposure` for the attributes and methods consumed
    downstream: ``_06_h5_export`` reads ``data``; the ``view_05`` viewer and
    :func:`mejiro.analysis.snr_calculation.get_snr` read ``data`` / ``lens_data`` /
    ``source_data`` / ``exposure_time`` / :meth:`get_snr`. Not suitable for
    re-running detector effects -- the galsim ``Image`` objects and per-effect
    noise components are not persisted. ``synthetic_image`` is ``None``, matching
    a full :class:`Exposure` round-tripped through :meth:`Exposure.__getstate__`;
    consumers reload the SyntheticImage from step 04 when they need it.
    """

    def __init__(self, data, meta, lens_data=None, source_data=None):
        self.data = data
        self.lens_data = lens_data
        self.source_data = source_data
        self.synthetic_image = None

        self.band = meta.get('band')
        self.instrument_name = meta.get('instrument_name')
        self.num_pix = meta.get('num_pix')
        self.pixel_scale = meta.get('pixel_scale')
        self.exposure_time = meta.get('exposure_time')
        self.engine = meta.get('engine')
        lens_meta = meta.get('lens') or {}
        self.name = lens_meta.get('name')
        self.z_lens = lens_meta.get('z_lens')
        self.z_source = lens_meta.get('z_source')
        self._meta = meta  # retained for debugging / introspection

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            data = np.asarray(f['data'])
            meta_bytes = f['meta'].tobytes()
            lens_data = np.asarray(f['lens_data']) if 'lens_data' in f else None
            source_data = np.asarray(f['source_data']) if 'source_data' in f else None
        meta = json.loads(meta_bytes.decode('utf-8'))
        schema_version = meta.get('schema_version')
        if schema_version != EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported lightweight schema_version={schema_version!r} "
                f"(expected {EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION}) in {path}"
            )
        return cls(data=data, meta=meta, lens_data=lens_data, source_data=source_data)

    @classmethod
    def from_npy(cls, path):
        """Data-only shim for a bare romanisim ``.npy`` cutout (no metadata/pieces)."""
        return cls(data=np.asarray(np.load(path)), meta={})

    def get_snr(self, snr_per_pixel_threshold=1):
        return get_snr(self, snr_per_pixel_threshold=snr_per_pixel_threshold)[0]

    def plot(self, show_snr=False, savepath=None):
        """Visualize the exposure (log10 counts). Mirrors :meth:`Exposure.plot`,
        reading band/instrument/exposure-time/lens scalars from stored metadata
        instead of ``self.synthetic_image``."""
        import matplotlib.pyplot as plt

        plt.imshow(np.log10(self.data), origin='lower')
        title = f'{self.name} (' + r'$z_{l}=$' + f'{self.z_lens:.2f}, ' + r'$z_{s}=$' + f'{self.z_source:.2f}' + f')\n{self.instrument_name} {self.band}, {self.exposure_time} s'
        if show_snr:
            snr = self.get_snr()
            title += f'\nSNR: {snr:.2f}'
        plt.title(title)
        cbar = plt.colorbar()
        cbar.set_label(r'log$_{10}$(Counts)')
        plt.xlabel('x [Pixels]')
        plt.ylabel('y [Pixels]')
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()
