import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pydicom import dcmread
import logging


def get_filenames(path):
    filenames = []
    for dirpath, _, files in os.walk(path):
        filenames += [os.path.join(dirpath, file) for file in files]
    return filenames


def get_dicom_meta(path):
    try:
        ds = dcmread(path)
    except Exception as e:
        print(f"Error reading {path}:", e)
        return None, None
    ds_meta = {'PatientID': ds.get('PatientID'),
               'PatientSex': ds.get('PatientSex'),
               'StudyDescription': ds.get('StudyDescription'),
               'SeriesDescription': ds.get('SeriesDescription'),
               'StudyInstanceUID': ds.get('StudyInstanceUID'),
               'SeriesInstanceUID': ds.get('SeriesInstanceUID'),
               'AcquisitionNumber': ds.get('AcquisitionNumber'),
               'InstanceNumber': ds.get('InstanceNumber'),
               'Modality': ds.get('Modality'),
               'StudyDate': ds.get('StudyDate'),
               'Rows': ds.get('Rows'),
               'Columns': ds.get('Columns'),
               'SliceLocation': ds.get('SliceLocation'),
               'Path': path
               }
    return ds_meta, ds


def create_meta_df(data_dir):
    paths = get_filenames(data_dir)
    meta_data = []
    for path in paths:
        meta_file, _ = get_dicom_meta(path)
        if meta_file != None:
            meta_data.append(meta_file)
    return pd.DataFrame(meta_data)


def display_dicom_file(path):
    meta_dict, ds = get_dicom_meta(path)
    if meta_dict != None:
        for key, value in meta_dict.items():
            print(f'{key}: {value}')
        # plot the image using matplotlib
        plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    # LOGFILENAME = '../logs/detector.log'
    # logging.basicConfig(filename=LOGFILENAME, level=logging.DEBUG)
    # DATA_DIR = '../data'
    # METADATA_FILENAME = 'metadata.pkl'
    DATA_DIR = '../data/nbia_data'
    METADATA_FILENAME = 'nbia_metadata.pkl'
    # paths = get_filenames(DATA_DIR)

    df = create_meta_df(DATA_DIR)

    with open(METADATA_FILENAME, 'wb') as f:
        pickle.dump(df, f)
