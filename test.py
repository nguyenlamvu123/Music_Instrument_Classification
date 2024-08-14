import joblib, os, sklearn, librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

from feature_engineering import *
from config import *


def Evaluate_model(y_true, y_pred):
    # site: https://machinelearningcoban.com/2017/08/31/evaluation/
    labels = np.array(y_true)
    print('test accuracy = ', accuracy_score(labels, y_pred), ' %')

    print(classification_report(labels, y_pred))

    cnf_matrix = confusion_matrix(labels, y_pred)
    print('Confusion matrix:\n', cnf_matrix)
    print('\nAccuracy:', np.diagonal(cnf_matrix).sum() / cnf_matrix.sum())


def draw3Dplotfor3feat():
    # https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py
    pass  # https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python


def main(PATH=None, testflag: bool = False) -> dict or None :
    if PATH is None:  # run test
        testflag = not testflag
        PATH = librosa.util.find_files(Test_path.data_path)
    labels = []
    samples = []
    for p in PATH:
        labels.append(p.split(os.sep)[-1].split('_')[0])
        sample, sr = librosa.load(p, sr=22050, duration=4.0)
        samples.append(sample)

    data = np.array([extract_feature(sample) for sample in samples])
    data = scaler.fit_transform(data)

    clf = joblib.load(Model.NAME)
    test_Y_hat = clf.predict(data)

    jso = None
    if testflag:  # run test
        Evaluate_model(labels, test_Y_hat)
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
    PATH = librosa.util.find_files('./dataset/valid')
    main(PATH, True)
