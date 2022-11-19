import tifffile
import numpy as np
from sdtfile import SdtFile
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def load_sdt(sdt_path, target_channel:int, num_channels:int = 3, im_dim:int = 512, num_bins:int = 256, plot:bool = False):
    ''' 
    Return desired channel from .sdt file as a numpy array
    Parameters
    ----------
    sdt_path : str or Path to .sdt file
    target_channel : int; desired channel number. NOT 0 indexed
    num_channels : int; number of channels in .sdt file. Default is 3
    im_dim : int; image xy dimensions. Assumes square image, Default = 512
    num_bins : int; number of bins in .sdt file. Default = 256
    '''
    sdt = SdtFile(sdt_path)
    data = sdt.data[0]
    data = data.reshape(num_channels, im_dim, im_dim, num_bins)
    data = data[target_channel - 1]
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(np.max(data, axis=2), cmap='gray'); ax.set_title(f'Channel {target_channel}'); ax.axis('off')
        plt.show()
    return data

def load_irf(irf_path, target_channel:int, num_channels:int = 3, im_dim:int = 512, num_bins:int = 256, plot:bool = False):
    ''' 
    Return instrument response function as a numpy array. If desired, plot the IRF and the region used to generate it.
    Parameters
    ----------
    irf_path : str or Path to .sdt file
    target_channel : int; desired channel number. NOT 0 indexed
    num_channels : int; number of channels in .sdt file. Default is 3
    im_dim : int; image xy dimensions. Assumes square image, Default = 512
    num_bins : int; number of bins in .sdt file. Default = 256
    '''
    irf_data = load_sdt(irf_path, target_channel=target_channel, num_channels=num_channels, im_dim=im_dim, num_bins=num_bins)
    summed_image = np.sum(irf_data, axis=2)
    otsu_threshold = threshold_otsu(summed_image)
    masked_region = summed_image > otsu_threshold
    masked_data = irf_data[masked_region]
    irf_summed = np.sum(masked_data, axis=0)
    if plot:
        x_axis = np.arange(0, 12.5, 12.5/num_bins)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(summed_image, cmap='gray'); ax1.set_title('Summed Image'); ax1.axis('off')
        ax2.imshow(masked_region); ax2.set_title('Masked Region'); ax2.axis('off')
        ax3.plot(x_axis, irf_summed); ax3.set_title('IRF'); ax3.set_xlabel('Time (ns)'); ax3.set_ylabel('Counts')
        fig.tight_layout()
        plt.show()
    return irf_summed

def bin_flim_data(data):
    ''' 
    Spatially bin time-resolved FLIM data to increase curve fitting ability.
    Parameters
    ----------
    data : numpy.ndarray
    '''
    if not data.ndim == 3:
        print("sorry, we're only doing single channels at the moment")
        return
    bin_num = 2
    kernel = np.ones((bin_num, bin_num,1))
    binned = convolve(data, kernel, mode='reflect')
    return binned

def load_asc(asc_path) -> np.ndarray:
    ''' 
    Loads .asc file as a numpy array
    Parameters
    ----------
    asc_path : str or Path to .asc file
    '''
    return np.loadtxt(asc_path)



