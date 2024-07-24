import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import hydra


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local modules
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util
    from mejiro.lenses import lens_util

    # set top directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'f_sub')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    detectable_lenses = lens_util.get_detectable_lenses(config.machine.pipeline_dir, limit=None, with_subhalos=True)
    print(f'Number of detectable lenses: {len(detectable_lenses)}')

    best_snr = [l for l in detectable_lenses if l.snr > 50]
    print(f'Number of high SNR lenses: {len(best_snr)}')

    num_pix = 91
    side = 10.01

    def plot_kappas(lens, sb, macrolens_kappa, subhalo_kappa, num_pix):
        f, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        ax[0].imshow(sb, norm=colors.LogNorm())
        ax[1].imshow(macrolens_kappa, cmap='bwr', norm=colors.LogNorm())
        ax[2].imshow(np.log10(subhalo_kappa), cmap='binary')
        ax[3].imshow(subhalo_kappa, cmap='bwr', vmin=-np.max(subhalo_kappa), vmax=np.max(subhalo_kappa))

        ax[0].set_title('Surface Brightness (log10)')
        ax[1].set_title('Macrolens Kappa (log10)')
        ax[2].set_title('Subhalo Kappa (log10)')
        ax[3].set_title('Subhalo Kappa')

        # overplot subhalos
        coords = lens_util.get_coords(num_pix=num_pix, delta_pix=0.11)
        for halo in lens.realization.halos:
            if halo.mass > 1e8:
                ax[2].plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='r', alpha=0.5)
            elif halo.mass > 1e7:
                ax[2].plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='g', alpha=0.25)
            else:
                ax[2].plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='b', alpha=0.1)

        for a in ax: a.axis('off')
        plt.savefig(os.path.join(save_dir, f'kappas_{lens.uid}.png'))
        plt.close()

    def annular_mask(dimx, dimy, center, r_in, r_out):
        Y, X = np.ogrid[:dimx, :dimy]
        distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        return (r_in <= distance_from_center) & \
        (distance_from_center <= r_out)
    
    def plot_f_sub(masked_kappa_subhalos, masked_kappa_macro, numerator, denominator, f_sub):
        f, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].imshow(masked_kappa_subhalos, cmap='bwr')
        ax[1].imshow(masked_kappa_macro, cmap='bwr')
        ax[0].set_title(f'{numerator:.2f}')
        ax[1].set_title(f'{denominator:.2f}')

        plt.suptitle(r'$f_{\mathrm{sub}}$ = ' + f'{f_sub:.6f}')

        for a in ax: a.axis('off')
        plt.savefig(os.path.join(save_dir, f'f_sub_{lens.uid}.png'))
        plt.close()

    # things to save
    einstein_radii = []
    f_subs = []

    for lens in tqdm(detectable_lenses):
        # generate synthetic image (surface brightness)
        sb = lens.get_array(num_pix=num_pix, side=side, band='F129')

        # total_kappa = lens.get_total_kappa(num_pix=num_pix, side=side)
        macrolens_kappa = lens.get_macrolens_kappa(num_pix=num_pix, side=side)
        subhalo_kappa = lens.get_subhalo_kappa(num_pix=num_pix, side=side)

        plot_kappas(lens, sb, macrolens_kappa, subhalo_kappa, num_pix)

        einstein_radius = lens.get_einstein_radius()
        einstein_radii.append(einstein_radius)

        r_in = (einstein_radius - 0.2) / 0.11  # units of pixels
        r_out = (einstein_radius + 0.2) / 0.11  # units of pixels

        mask = annular_mask(*sb.shape, (sb.shape[0]//2, sb.shape[1]//2), r_in, r_out)

        masked_kappa_subhalos = np.ma.masked_array(subhalo_kappa, mask=~mask)
        masked_kappa_macro = np.ma.masked_array(macrolens_kappa, mask=~mask)

        numerator = masked_kappa_subhalos.compressed().sum()
        denominator = masked_kappa_macro.compressed().sum()
        f_sub = numerator / denominator
        f_subs.append(f_sub)

        plot_f_sub(masked_kappa_subhalos, masked_kappa_macro, numerator, denominator, f_sub)

    # save
    np.save(os.path.join(save_dir, 'einstein_radii.npy'), np.array(einstein_radii))
    np.save(os.path.join(save_dir, 'f_subs.npy'), np.array(f_subs))


if __name__ == '__main__':
    main()
