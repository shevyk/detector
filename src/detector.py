import os
from pydicom import dcmread
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pickle


def get_filenames(path):
    dicom_files = []
    for dirpath, dirnames, files in os.walk(path):
        dicom_files += [os.path.join(dirpath, file) for file in files]
    return dicom_files


def get_meta(path):
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
               'Path': path}
    return ds_meta


def display_dicom_file(path):
    ds = dcmread(path)
    print()
    print(f"File path........: {path}")
    print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
    print()

    print(f"Patient ID.........: {ds.PatientID}")
    print(f"Patient Sex........: {ds.PatientSex}")
    print(f"Study Description..: {ds.StudyDescription}")
    print(f"Series Description.: {ds.SeriesDescription}")
    print(f"Modality...........: {ds.Modality}")
    print(f"Study Date.........: {ds.StudyDate}")
    print(f"Image size.........: {ds.Rows} x {ds.Columns}")
    print(f"Pixel Spacing......: {ds.PixelSpacing}")

    # use .get() if not sure the item exists, and want a default value if missing
    print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

    # plot the image using matplotlib
    plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    data_dir = '../data'
    paths = get_filenames(data_dir)
    meta_data = []
    
    for path in paths:
        meta_file = get_meta(path)
        if meta_file != None:
            meta_data.append(meta_file)
    
    df = pd.DataFrame(meta_data)
    
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(df, f)

