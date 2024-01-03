import copy

@export
def magnitude2amplitude(light_model_class, kwargs_light_mag, magnitude_zero_point):
    """Translates astronomical magnitudes to lenstronomy linear 'amp' parameters for
    LightModel objects.

    :param light_model_class: LightModel() class instance
    :param kwargs_light_mag: list of light model parameter dictionary with 'magnitude'
        instead of 'amp'
    :param magnitude_zero_point: magnitude zero point
    :return: list of light model parameter dictionary with 'amp'
    """
    kwargs_light_amp = copy.deepcopy(kwargs_light_mag)
    if kwargs_light_mag is not None:
        for i, kwargs_mag in enumerate(kwargs_light_mag):
            kwargs_new = kwargs_light_amp[i]
            del kwargs_new["magnitude"]
            cps_norm = light_model_class.total_flux(
                kwargs_list=kwargs_light_amp, norm=True, k=i
            )[0]
            magnitude = kwargs_mag["magnitude"]
            cps = magnitude2cps(magnitude, magnitude_zero_point=magnitude_zero_point)
            amp = cps / cps_norm
            kwargs_new["amp"] = amp
    return kwargs_light_amp
