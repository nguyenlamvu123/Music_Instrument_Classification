import sys
import os
import numpy as np
import pandas as pd
import librosa
from sklearn import svm
import joblib

from config import Model, CreateDataset, debug, batch_size
from feature_engineering import readdata, timer

csv_name = CreateDataset.Name


def define_model(C=1.0, gamma=0.02, degree=3, coef0=0.0, ):
    print(f'C={C}, gamma={gamma}, degree={degree}, coef0={coef0}')
    return svm.SVC(
        C=C,
        cache_size=200,
        class_weight=None,
        coef0=coef0,
        decision_function_shape='ovr',
        degree=degree,
        gamma=gamma,
        kernel='linear',  # 'rbf',  #
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False
    )

def main(audios_numpy, labels):
    # Load data
    n_samples = audios_numpy.shape[0]
    n_batches = n_samples // batch_size

    clf = define_model()

    for i in range(n_batches):
        print(f"{i + 1}/{n_batches}")
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_y = labels[start_idx:end_idx]
        assert len(set(batch_y)) > 1
        piece = audios_numpy[start_idx:end_idx]
        batch_X = readdata(piece)  # for audio in tqdm(piece):

        dataset_pandas = pd.DataFrame(batch_X)
        dataset_pandas["instruments"] = batch_y
        dataset_pandas.to_csv(csv_name, mode='a', index=False, header=False)

        linear_svm.fit(batch_X, batch_y)
        joblib.dump(linear_svm, Model.NAME)

    if n_samples % batch_size != 0:
        piece = audios_numpy[n_batches * batch_size:]
        remaining_y = labels[n_batches * batch_size:]
        if len(set(remaining_y)) > 1:
            remaining_X = readdata(piece)  # for audio in tqdm(piece):

            dataset_pandas = pd.DataFrame(remaining_X)
            dataset_pandas["instruments"] = remaining_y
            dataset_pandas.to_csv(csv_name, mode='a', index=False, header=False)

            linear_svm.fit(remaining_X, remaining_y)
            joblib.dump(linear_svm, Model.NAME)
