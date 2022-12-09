from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

import pydicom

from skimage import transform


from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pydicom import dcmread
import pydicom
import PIL.ImageOps  
from skimage import exposure
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms



def read_image(dcm):
    
    img = dcm.pixel_array.astype(float)
    img  = (np.maximum(img,0) / img.max()) * 255.0
    img = np.uint8(img)
    img = Image.fromarray(img)
    if is_inverse(dcm) == True:
        img = PIL.ImageOps.invert(img)
    return img

seg_transform = A.Compose(
    [A.Normalize(
              mean=[0.0,0.0,0.0],
              std=[1.0,1.0,1.0],
              max_pixel_value=255.0),
        ToTensorV2()])

# crop image        
def crop_image(image):
    width, height = image.size
    left = 10
    top = 50
    right = width - 10
    bottom = height -50
    return np.array(image.crop((left, top, right, bottom)))

# resize image with padding
def resize_image(im, dim):
    im = Image.fromarray(im)
    dims = [im.size[0], im.size[1]]
    factor = dim/max(dims)
    resized_im = im.resize((round(im.size[0]*factor), round(im.size[1]*factor)))
    # Setting the points for cropped image
    left = -1*((dim-resized_im.size[0])/2)
    top = -1*((dim- resized_im.size[1])/2)
    right = (dim+resized_im.size[0])/2
    bottom = (dim+resized_im.size[1])/2
    resized =  resized_im.crop((left, top, right, bottom))
    return np.array(resized)

# detect if image inverted
def is_inverse(ds):
    return ds.get('PresentationLUTShape', False) and ds.PresentationLUTShape == 'INVERSE' or ds.PhotometricInterpretation == 'MONOCHROME1'

# apply transform for segmentation model
def seg_img_transform(image, transform , device):
    image = np.array(image)
    augmentations = transform(image = image)
    image = augmentations['image']
    return image.to(device).unsqueeze(0)


# enhance image brightness for preprocessing
def intensity_transfer(image):
    x_mean = np.mean(np.array(image))
    x_std = np.std(np.array(image))

    if x_mean-x_std > 20:
        mu = x_mean-30
        sigma = x_std+30
    elif x_mean-x_std < 20:
        mu = x_mean +10 
        sigma = x_std +10
    elif x_std > x_mean:
        mu = x_mean+40
        sigma = mu
            
    height, width= image.shape
    for i in range(0,height):
        for j in range(0,width):
            x = image[i,j]
            x = ((x-x_mean)*(sigma/x_std))+mu
            x = round(x.item())
            # boundary check
            x = 0 if x<0 else x    
            x = 250 if x>250 else x
            image[i,j] = x
    # mean, std = np.mean(image), np.std(image)
    return image 

# enhance image brightness for postprocessing
def intensity_transfer2(image, mu, sigma):
    x_mean = np.mean(np.array(image))
    x_std = np.std(np.array(image))
    height, width= image.shape
    for i in range(0,height):
        for j in range(0,width):
            x = image[i,j]
            x = ((x-x_mean)*(sigma/x_std))+mu
            x = round(x.item())
            # boundary check
            x = 0 if x<0 else x    
            x = 250 if x>250 else x
            image[i,j] = x
    # mean, std = np.mean(image), np.std(image)
    return image 

# increase contrast for postprocessing
def contrast_stretching(image, a,b):
    p2, p98 = np.percentile(image, (a, b))
    return exposure.rescale_intensity(image, in_range=(p2, p98))




def series_selection(input_directory: Path) -> List[pydicom.Dataset]:
    """Perform series selection and return selected datasets.

    All .dcm files in the input directory and its sub-directories
    will be considered. The first series found with Modality CT
    and an axis-aligned axial orientation will be used.

    Parameters
    ----------
    input_directory: pathlib.Path
        Input directory containing .dcm files.

    Returns
    -------
    List[pydicom.Dataset]
        Unsorted list of all the datasets in the selected series.

    """
    series_dict = defaultdict(list)

    # Read in all DICOM files, sorted into series by UID
    for f in input_directory.glob("**/*.dcm"):
        dcm = pydicom.dcmread(f)
        series_dict[dcm.SeriesInstanceUID].append(dcm)

    # Loop over series. Choose the first axial CT series
    for series_uid, dcm_list in series_dict.items():
        dcm = dcm_list[0]
        # Must be a CT with ORIGINAL Image Type
        if dcm.Modality == "CR" or dcm.Modality == 'DX':
            for dcm in dcm_list:
                dcm.StudyID = "none"
            return dcm_list
    else:
        raise RuntimeError("No axial X-Ray series found.")

