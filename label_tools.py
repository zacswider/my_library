import numpy as np


def get_label_vals(arr: np.ndarray) -> np.ndarray:
    """Return numpy array containing label numbers in arr.
    Args:
        arr (np.ndarray): Array of labels. Each label is a continuous region of the same value.
    
    Returns:
        np.ndarray: Array of label numbers.
    """
    return np.nonzero(np.unique(arr))[0]  # type: ignore


def remove_large_objects(labels_array: np.ndarray, max_size: int) -> np.ndarray:
    ''' 
    Remove all objects in a mask above a specific threshold
    '''
    out = np.copy(labels_array)
    component_sizes = np.bincount(labels_array.ravel()) 
    too_big = component_sizes > max_size
    too_big_mask = too_big[labels_array]
    out[too_big_mask] = 0
    return out


def return_points(labels_array: np.ndarray, label_ID: int) -> np.ndarray:
    '''
    Return the points in a mask that belong to a specific label
    ---
    Parameters:
    labels_array: np.ndarray an ndArray of labels
    label_ID: int the label ID of the label whos points you want to calculate
    ---
    Returns:
    points: np.ndarray an ndArray of shape (n,3) where n is the number of points in the label
    and dim1 is the x,y,z coordinates of the points
    '''
    points = np.column_stack(np.where(labels_array == label_ID))
    return points


