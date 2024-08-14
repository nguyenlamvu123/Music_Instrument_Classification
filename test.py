import joblib, os, sklearn, librosa
import numpy as np

from feature_engineering import *
from config import *

def main(PATH=None) -> dict or None :
    if PATH is None:  # run test
        PATH = librosa.util.find_files(Test_path.data_path)
    labels = []
    samples = []
    for p in PATH:
        labels.append(p.split(os.sep)[-1].split('_')[0])
        sample, sr = librosa.load(p, sr=22050, duration=4.0)
        samples.append(sample)

    data = np.array([extract_feature(sample) for sample in samples])

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data)

    clf = joblib.load(Model.NAME)
    test_Y_hat = clf.predict(data)

    jso = None
    if PATH is None:  # run test
        accuracy = np.sum((test_Y_hat == labels)) / 200.0 * 100.0
        print('test accuracy = ' + str(accuracy) + ' %')
    else:  # method is calles from gradio
        result = list(test_Y_hat)
        jso: dict or None = dict()
        assert len(result) == len(PATH)
        res_set: set = set(result)
        for key in res_set:
            jso[key] = list()
            for i in range(len(result)):
                if result[i] == key:
                    jso[key].append(PATH[i].split(os.sep)[-1])
    return jso


if __name__ == '__main__':
    main()
