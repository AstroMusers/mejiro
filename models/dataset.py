class Dataset:

    def __init__(self, 
                 data_set_name, 
                 target_name, 
                 ra, 
                 dec, 
                 start_time, 
                 stop_time,
                 actual_duration,
                 instrument, 
                 aper, 
                 spec, 
                 central_wavelength, 
                 release_date, 
                 preview_name, 
                 filename, 
                 data_uri, 
                 cutout, 
                 peak_value,
                 fits_filepath,
                 cutout_filepath):
        self.data_set_name = data_set_name
        self.target_name = target_name
        self.ra = ra
        self.dec = dec
        self.start_time = start_time
        self.stop_time = stop_time
        self.actual_duration = actual_duration
        self.instrument = instrument
        self.aper = aper
        self.spec = spec
        self.central_wavelength = central_wavelength
        self.release_date = release_date
        self.preview_name = preview_name
        self.filename = filename
        self.data_uri = data_uri
        self.cutout = cutout
        self.peak_value = peak_value
        self.fits_filepath = fits_filepath
        self.cutout_filepath = cutout_filepath

    def set_filename(self, filename):
        self.filename = filename

    def set_data_uri(self, data_uri):
        self.data_uri = data_uri

    def set_fits_filepath(self, fits_filepath):
        self.fits_filepath = fits_filepath

    def set_cutout_filepath(self, cutout_filepath):
        self.cutout_filepath = cutout_filepath

    def __str__(self):
        return self.target_name + ': ' + self.data_set_name
    