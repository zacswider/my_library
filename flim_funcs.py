import tifffile
import numpy as np
from sdtfile import SdtFile
from pathlib import Path 
import tkinter as tk
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def load_sdt(sdt_path, target_channel:int, num_channels:int = 3, im_dim:int = 512, num_bins:int = 256, plot:bool = False) -> np.ndarray:
    ''' 
    Return desired channel from .sdt file as a numpy array
    Parameters
    ----------
    sdt_path : str or Path to .sdt file
    target_channel : int; desired channel number. NOT 0 indexed
    num_channels : int; number of channels in .sdt file. Default is 3
    im_dim : int; image xy dimensions. Assumes square image, Default = 512
    num_bins : int; number of bins in .sdt file. Default = 256

    Returns
    -------
    data : numpy.ndarray of shape (im_dim, im_dim, num_bins)
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

def load_irf(irf_path, target_channel:int, num_channels:int = 3, im_dim:int = 512, num_bins:int = 256, plot:bool = False) -> np.ndarray:
    ''' 
    Return instrument response function as a numpy array. If desired, plot the IRF and the region used to generate it.
    Parameters
    ----------
    irf_path : str or Path to .sdt file
    target_channel : int; desired channel number. NOT 0 indexed
    num_channels : int; number of channels in .sdt file. Default is 3
    im_dim : int; image xy dimensions. Assumes square image, Default = 512
    num_bins : int; number of bins in .sdt file. Default = 256

    Returns
    -------
    irf : numpy.ndarray of shape (num_bins,)
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

def bin_flim_data(data: np.ndarray) -> np.ndarray:
    ''' 
    Spatially bin time-resolved FLIM data to increase curve fitting ability.
    Parameters
    ----------
    data : numpy.ndarray. The data to be binned.

    Returns
    -------
    binned_data : numpy.ndarray. The binned data.
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

    Returns
    -------
    data : numpy.ndarray
    '''
    return np.loadtxt(asc_path)

def batch_asc_files_to_tif(in_path = None) -> None:
    ''' 
    Batch converts out put of SPCImage to tif files. Input is a path to the folder containing the asc files,
    typically there is a separate .asc file for each parameter (e.g. a1, a2, t1, t2, etc.)
    Parameters
    ----------
    in_path : String or Path | optional. If None, will prompt user to select folder. The default is None.
    
    Returns
    -------
    None.
    '''
    if in_path is None:
        root = tk.Tk()
        root.withdraw()
        in_path = tk.filedialog.askdirectory()

    in_path = Path(in_path)
    file_names = [f.stem for f in in_path.glob('*.asc')]

    # get unique base file names
    base_names = [f.rsplit('_', maxsplit=1)[0] for f in file_names]
    base_names = np.unique(base_names)

    # get unique param names for each base name
    for base_name in tqdm(base_names):
        avoid = ['offset', 'scatter', 'shift']
        params = [f.stem.rsplit('_', maxsplit=1)[-1] for f in in_path.glob(f'{base_name}_*.asc')]
        params = [p for p in params if p not in avoid]
        params = np.unique(params)
        
        # load the first asc file to get the shape
        x_shape, y_shape = load_asc(in_path / f'{base_name}_{params[0]}.asc').shape
        depth = len(params)
        asc_array = np.ones(shape=(depth, x_shape, y_shape))

        # combine the asc files into a single array
        for i, params_name in enumerate(params):
            asc_array[i] = load_asc(in_path / f'{base_name}_{params_name}.asc')
        
        # save the array as a tif
        out_path = in_path / 'concatenated_files' 
        out_path.mkdir(exist_ok=True)
        tifffile.imwrite(out_path / f'{base_name}.tif', asc_array.astype(np.float32), imagej=True, metadata={'axes': 'CYX'})

        # save the channel order as text file
        with open(out_path / f'{base_name}_channel_order.txt', 'w') as f:
            for i, params_name in enumerate(params):
                f.write(f'{i} = {params_name}\n')

