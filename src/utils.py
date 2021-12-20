import os
from pydicom import dcmread
import pandas as pd
import pickle


def get_filenames(path):
    dicom_files = []
    for dirpath, dirnames, files in os.walk(path):
        dicom_files += [os.path.join(dirpath, file) for file in files]
    return dicom_files


def get_dicom_meta(path):
    try:
        ds = dcmread(path)
    except Exception as e:
        print(f"Error reading {path}:", e)
        return
    ds_meta = {'PatientID': ds.PatientID,
               'PatientSex': ds.PatientSex,
               'StudyDescription': ds.StudyDescription,
               'SeriesDescription': ds.SeriesDescription,
               'Modality': ds.Modality,
               'StudyDate': ds.StudyDate,
               'Rows': ds.Rows,
               'Columns': ds.Columns,
               'SliceLocation': ds.SliceLocation,
               'Path': path
               }
    return ds_meta

def create_meta_df(data_dir):
    paths = get_filenames(data_dir)
    meta_data = []
    
    for path in paths:
        meta_file = get_dicom_meta(path)
        if meta_file != None:
            meta_data.append(meta_file)
    
    return pd.DataFrame(meta_data)


if __name__ == '__main__':
    data_dir = '../data'
    filename = 'metadata.pkl'
    
    df = create_meta_df(data_dir)
    
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    