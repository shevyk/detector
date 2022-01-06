from pydicom import dcmread
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

def get_modalities_from_labels(labels, modalities):
    modality_type = pd.DataFrame(['CT']*len(labels))
    for modality in modalities:
        modality_type[labels.str.contains(modality, case=False)] = modality
    return modality_type

def get_body_region_from_labels(labels, region_dict):
    body_region = pd.DataFrame(['np.nan']*len(labels))
    for key, value in region_dict.items():
        body_region[labels.str.contains(key, case=False)] = value
    return body_region

def clean_the_data(df, modalities, region_dict):
    df = df.dropna(subset=['StudyDescription'])
    # # Drop whole body scans
    # df = df[df.StudyDescription != 'PET^1PETCT_WholeBody (Adult)']
    df['Modality_Type'] = get_modalities_from_labels(df.StudyDescription, modalities)
    df['Body_Region'] = get_body_region_from_labels(df.StudyDescription, region_dict)
    df = df.dropna(subset=['Body_Region'])
    df['Class'] = df['Modality_Type'] + '_' + df['Body_Region']
    return df

def sample_from_patient(meta_df, n):
    '''
    Parameters
    ----------
    meta_df: DataFrame
            dataframe containing Dicom files metadata
    n : int
        number entries to sample from each patient
        if n > number of patient samples, then the min 
        number of patiet entries will be returned

    Returns
    -------
    DataFrame

    '''
    g = meta_df.groupby("PatientID")
    n = min(g.PatientID.count().min(), n)
    return meta_df.groupby("PatientID").sample(n).reset_index()

def rescale_hu(image, slope, intercept):
    '''
    Rescale DICOM file image to the Hounsfield scale
    using the following linear transformation:
    rescaled pixel = pixel * slope + intercept

    Parameters
    ----------
    image : TYPE numpy.ndarray
        DESCRIPTION.
    slope : TYPE, optional
        DESCRIPTION. DICOM file RescaleSlope attribute.
    intercept : TYPE, optional
        DESCRIPTION. DICOM file RescaleIntercept attribute.

    Returns
    -------
    image : TYPE numpy.ndarray
        DESCRIPTION. rescaled image

    '''
    return image * slope + intercept


def clip_hu(image, min_val=0, max_val=2000):
    '''
    clip image pixel values to min_val and max_val

    Parameters
    ----------
    image : TYPE numpy.ndarray
        DESCRIPTION.
    min_val : TYPE, optional
        DESCRIPTION. The default is 0.
    max_val : TYPE, optional
        DESCRIPTION. The default is 2000.

    Returns
    -------
    image : TYPE numpy.ndarray
        DESCRIPTION. clipped image

    '''
    image[image < min_val] = min_val
    image[image > max_val] = max_val
    return image


def torchvision_transforms(image):
    '''
    Prepares image for ingestion by torchvision pretrained model 

    Parameters
    ----------
    image : numpy.ndarray
        DESCRIPTION.

    Returns
    -------
    TYPE torch tensor
        DESCRIPTION.

    '''
    PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]
                                          )
    return image_transforms(PIL_image)


def get_transformed_image(path):
    dicom_file = dcmread(path)
    image = dicom_file.pixel_array
    image = rescale_hu(image, dicom_file.RescaleSlope,
                       dicom_file.RescaleIntercept)
    image = clip_hu(image)
    image = torchvision_transforms(image)
    return image


