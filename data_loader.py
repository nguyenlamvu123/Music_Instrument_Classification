import sys
import os
import numpy as np
import librosa
import pandas as pd
import sklearn
from tqdm import tqdm

from feature_engineering import *
from config import CreateDataset

data_path = CreateDataset.data_path
csv_name = CreateDataset.Name


# file load
def get_sampels(data_set='train'):
    audios = []
    labels = []
    path_of_audios = librosa.util.find_files(data_path + data_set)
    for audio in path_of_audios:
        labels.append(audio.split(os.sep)[-1].split('_')[0])
        audios.append(audio)
    audios_numpy = np.array(audios)
    return audios_numpy, labels


def data2csv():  # data to csv
    is_created = False
    audios_numpy, labels = get_sampels(data_set='train')
    for samples in audios_numpy:
        row = extract_feature(samples)
        if not is_created:
            dataset_numpy = np.array(row)
            is_created = True
        elif is_created:
            dataset_numpy = np.vstack((dataset_numpy, row))

    dataset_numpy = scaler.fit_transform(dataset_numpy)

    dataset_pandas = pd.DataFrame(dataset_numpy)
    dataset_pandas["instruments"] = labels
    dataset_pandas.to_csv(csv_name, index=False)


if __name__ == '__main__':
    data2csv()



