import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


class PandeiaOutput:
    def __init__(self, results):
        self.results = results

    def get_dicts(self):
        dicts = []
        for key in self.results.keys():
            dicts.append(key)

        return dicts

    def plot_model_flux_cube(self):
        flux = self.results['3d']['flux']

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(x=flux[0], y=flux[1], z=flux[2])

        return ax

    def get_snr(self):
        snr = self.results['2d']['snr']
        return np.flipud(snr)

    def get_image(self):
        image = self.results['2d']['detector']
        return np.flipud(image)

    def print_scalars(self):
        pprint(self.results['scalar'])
