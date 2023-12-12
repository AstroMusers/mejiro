import numpy as np
import matplotlib.pyplot as plt


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof



def twoD_ps(data=None,pix_size=0,rnge=100,shift=0,show_ps=False):
    """
    takes in a 2D array and returns the 2D FFT:
    inputs:
        ind_coords: n 2d arrays, whose average = coadd_coords
        coadd_coords: a singe 2d array
        pix_size: pixel size
        rnge: box size
        shift: offset from the halo center
        show_ps: whether you want to display the 2d power spectrum
    outputs:
        ind_ps_x: list of lists, where each list is a 2d power spectrum
        tot_ps: total 2D power spectrum after coadding all PS in ind_ps_x
        ky,ky: fft frequencies
    """
    if data is None:
        raise RuntimeError('No data was given!')

    ind_ps = []
    for i in data:
        ft = np.fft.fft2(i)
        ps2D = np.abs(ft)**2
        ind_ps.append(ps2D)

    A_pix = pix_size**2
    A_box = rnge**2
    norm = A_pix**2/A_box

    ind_ps_x = [norm*np.fft.fftshift(i) for i in ind_ps]
    tot_ps = np.mean(ind_ps_x,axis=0)
    tot_ps2 = np.log10(tot_ps)

    kx = 2*np.pi*np.fft.fftfreq(tot_ps.shape[0],d=pix_size)
    kx = np.fft.fftshift(kx)
    ky = 2*np.pi*np.fft.fftfreq(tot_ps.shape[1],d=pix_size)
    ky = np.fft.fftshift(ky)

    if show_ps == True:
        fig,(ax1,ax2) = plt.subplots(2,sharey=True)
        ax1.imshow(tot_ps,extent=[min(kx),max(kx),min(ky),max(ky)],interpolation='nearest')
        ax2.imshow(tot_ps2,extent=[min(kx),max(kx),min(ky),max(ky)],interpolation='nearest')
        plt.show()

    return ind_ps_x, tot_ps, kx, ky