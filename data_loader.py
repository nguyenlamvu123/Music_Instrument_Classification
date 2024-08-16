import os
import pandas as pd
from tqdm import tqdm

from feature_engineering import *

data_path = CreateDataset.data_path
csv_name = CreateDataset.Name


# file load
def get_sampels(data_set='train'):
    audios = []
    labels = []
    path_of_audios = librosa.util.find_files(data_path + data_set)
    enough_genre = list()
    gendic = dict()
    for audio in path_of_audios:
        label = audio.split(os.sep)[-1].split('_')[0]
        if label in enough_genre:
            continue
        if label in gendic:
            if gendic[label] >= 5500:
                print(label, '__', gendic[label])
                enough_genre.append(label)
                continue
            else:
                gendic[label] += 1
        else:
            gendic[label] = 0
        labels.append(label)
        y, _ = librosa.load(audio, sr=22050, duration=4.0)
        audios.append(y)
    audios_numpy = np.array(audios)
    return audios_numpy, labels

def main():
    is_created = False
    audios_numpy, labels = get_sampels(data_set='train')
    for samples in tqdm(audios_numpy):
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
    main()



