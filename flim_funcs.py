import tifffile
import numpy as np
from sdtfile import SdtFile
from pathlib import Path 
import tkinter as tk
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


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
    assert data.ndim == 3, "sorry, we're only doing single channels at the moment"
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
        in_path = tk.filedialog.askdirectory()                      # type: ignore

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


def find_base_file_names(p: Path):
    """Sort through a folder of .asc files and return a list of the base file names
    
    Args
        p (Path): Path to folder containing .asc files
    
    Returns
        unique_names (list): List of base file names
    """
    asc_files = list(p.glob('*.asc'))
    # remove anything with the word 'phasor' in it
    asc_files = [x for x in asc_files if 'phasor' not in x.name]
    base_names = [f.stem.rsplit('_', maxsplit = 1)[0] for f in asc_files]
    return list(np.unique(base_names))


def assign_group_to_names(file_names: list, group_names: dict, assignment: str = 'all') -> dict:
    """Given a list of file names and dictionary with unique identifiers for each group,
    assign a group to each file name and return assignments as a dictionary.
    
    Args
        file_names (list): List of file names
        group_names (dict): Dictionary with group names as keys and unique identifiers as values
        assignment (str): 'all' or 'any'. If 'all', a file name must contain all unique identifiers,
        if 'any', a file name may contain any unique identifier.
    
    Returns 
        group_assignments (dict): Dictionary with file names as keys and group names as values
    """
    assigned_names = {}
    for name in file_names:
        potential_group_matches = []
        for group_name, identifiers in group_names.items():
            if assignment == 'all':
                if any([x not in name for x in identifiers]):
                    continue
                potential_group_matches.append(group_name)
            elif assignment == 'any':
                if not any([x in name for x in identifiers]):
                    continue
                potential_group_matches.append(group_name)
        if len(potential_group_matches) == 1:
            assigned_names[name] = potential_group_matches[0]
        else:
            print(f'{len(potential_group_matches)} matches found')
            print(f'Could not assign {name} to a group')
            assigned_names[name] = 'no group found'
    return assigned_names


def measure_asc_files(base_file_paths: list, 
                      measurements: list, 
                      group_assignments: dict) -> pd.DataFrame:
    """Measure a list of files and return a pandas dataframe with the results
    
    Args
        base_file_paths (list): List of paths to files to be measured. NOTE: these should be full path to base
        file names, it is assumed that the full path will include a _{meas}.asc at the end.
        
    Returns
        df (pd.DataFrame): Pandas dataframe with results
    """
    data_collection = []
    for full_path in tqdm(base_file_paths):
        
        file_name = full_path.stem

        file_measurements = {
            'file name': file_name,
            'group': group_assignments[file_name]
        }

        for measurement in measurements:
            file_name = f'{full_path}_{measurement}.asc'
            asc_asarray = load_asc(file_name)
            mask = asc_asarray > 0
            file_measurements[f'mean {measurement}'] = np.mean(asc_asarray[mask])
            file_measurements[f'median {measurement}'] = np.median(asc_asarray[mask])
            file_measurements[f'std {measurement}'] = np.std(asc_asarray[mask])

        data_collection.append(file_measurements)
    
    return pd.DataFrame(data_collection)


def measure_asc_simple(path_to_asc_files: Path, group_names: dict, assignment_type: str, 
                       measurements: list) -> pd.DataFrame:
    """This is a simple function to streamline a simple analysis of a folder full of asc files.
    
    Args
        path_to_asc_files (Path): Path to the folder containing the asc files
        group_names (dict): Dictionary of group names and the unique identifiers for each group
        assignment_type (str): How to assign the group names to the asc files. Options are 'all' or 'any'. 
            'all' means that all the unique identifiers must be present in the file name to be assigned to the group.
            'any' means that any of the unique identifiers must be present in the file name to be assigned to the group.
        measurements (list[str]): List of asc file types to expect, for example ..._a1[%] or ..._t1
        measurement_type (str): What type of measurement to take. Options are 'mean', 'median', 'std'
    Returns
        pd.DataFrame: A dataframe containing the measurements for each group
    """
    unique_names = find_base_file_names(path_to_asc_files)
    assigned_groups = assign_group_to_names(unique_names, group_names, assignment = assignment_type)
    df = measure_asc_files(base_file_paths = [path_to_asc_files / name for name in unique_names], 
                           measurements = measurements, 
                           group_assignments = assigned_groups)
    return df


def visualize_simple(df: pd.DataFrame, group_names: list, measurements: list, 
                     measurement_type: str = 'mean', dpi: int = 100) -> None:
    """Simple function to visualize the results of a simple asc file analysis
    
    Args
        df (pd.DataFrame): Pandas dataframe containing the results of the analysis
        group_names (list): List of group names to be plotted
        measurements (list[str]): List of measurements to expect, for example ..._a1[%] or ..._t1
        measurement_type (str): What type of measurement to take. Options are 'mean', 'median', 'std'
        dpi (int): Dots per inch for the figure
    
    Returns
        None
    """
    accepted_measurements = ['a1[%]', 'a2[%]', 't1', 't2']
    measurements = [m for m in measurements if m in accepted_measurements]
    assert len(measurements) <= 4, 'Too many measurements to plot'
    fig, axes = plt.subplot_mosaic(mosaic = '''
                                        AB
                                        CD
                                        ''',
                                        figsize = (10, 10),
                                        dpi = dpi)
    for ax, meas in zip(axes.keys(), measurements):                                             # type: ignore 
        curr_ax = axes[ax]                                                                      # type: ignore                                            
        curr_ax.set_title(meas)
        sns.boxplot(x = 'group', y = f'{measurement_type} {meas}', data = df, ax = curr_ax)
        sns.scatterplot(x = 'group', y = f'{measurement_type} {meas}', data = df, ax = curr_ax, color = 'black')
        curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation = 45)
    fig.tight_layout()                                                                          # type: ignore
    plt.show()
    

