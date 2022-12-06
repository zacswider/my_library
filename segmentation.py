from cellpose import models
from cellpose.io import imread, save_masks
import os
from pathlib import Path


def load_cellpose_model(model_name: str, use_gpu: bool = True) -> tuple:
    """Load and return a pre-trained or custom cellpose model
    
    Args:
        model_name (str): Name of the model to load. Can be the name of a 
        pre-trained model or thhe name of a custom model.
    
    returns:
        (model type, model object): A tuple containing the model type ('pretrained' or 'custom') and the model object
    """
    path_to_custom_models = Path(f'C:/Users/{os.getlogin()}/.cellpose/models')
    assert path_to_custom_models.exists(), f'Path to custom models does not exist: {path_to_custom_models}'
    
    pretrained_models = [
        'cyto',
        'cyto2',
        'nuclei',
        'tissuenet',
        'livecell'  
    ]
    
    if model_name in pretrained_models:
        return 'pretrained', models.Cellpose(gpu=use_gpu, model_type=model_name)
    
    else:
        models_in_cust_model_dir = os.listdir(path_to_custom_models)
        assert model_name in models_in_cust_model_dir, "model name doesn't match either a pre-trained model or a custom model"
        return 'custom', models.CellposeModel(gpu=use_gpu, pretrained_model = str(path_to_custom_models / model_name))  # type: ignore
 
         
def apply_cellpose_model(file_path: Path, out_dir: Path, model_name: str, diameter: int = 30, use_gpu: bool = True, flow_threshold: float = 0.4) -> None:
    """Use a custom model to segment images in a directory.
    
    Args:
        in_dir (Path): Path to images to segment.
        out_dir (Path): Path to directory where segmented images will be saved.
        model_name (str): Name of custom model to use for segmentation.
        file_type (str, optional): File type of images to segment. Defaults to '.tif'.
        diameter (int, optional): Diameter of cell to segment. Defaults to 30.
        use_gpu (bool, optional): Whether to use GPU for segmentation. Defaults to True.
        flow_threshold (float, optional): Threshold for flow error. Defaults to 0.4.
    """
    file_path = Path(file_path)
    im = imread(file_path)
    assert im.ndim == 2, f'Only 2D images are supported, {im.ndim} dims found with shape {im.shape}'     # type: ignore
    model_type, model = load_cellpose_model(model_name, use_gpu = use_gpu)
    assert model_type in ['pretrained', 'custom'], 'Model type must be either "pretrained" or "custom"'
    if model_type == 'pretrained':
        masks, flows, styles, diameter = model.eval(im, diameter=diameter, channels=[0,0], flow_threshold=flow_threshold, do_3D=False)
    else:
        masks, flows, styles = model.eval(im, diameter=diameter, channels=[0,0], flow_threshold=flow_threshold)
    
    save_masks(images = im, flows = flows, masks = masks, file_names=file_path.name, savedir=out_dir)



