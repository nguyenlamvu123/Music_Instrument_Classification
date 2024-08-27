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


@timer
def main(PATH=None, testflag: bool = False, clf=None, jso: dict or None = None) -> dict or None :
    if PATH is None:  # run test
        testflag = True
        PATH = librosa.util.find_files(Test_path.data_path)
    if debug:
        PATH = PATH[:3]
    labels = []
    samples = []
    for p in PATH:
        labels.append(p.split(os.sep)[-1].split('_')[0])
        sample, sr = librosa.load(p, sr=22050, duration=4.0)
        samples.append(sample)

    data = np.array([extract_feature(sample) for sample in samples])
    data = scaler.fit_transform(data)

    if clf is None: clf = joblib.load(Model.NAME)
    test_Y_hat = clf.predict(data)

    if testflag:  # run test
        Evaluate_model(labels, test_Y_hat)
    else:  # method is calles from gradio
        result = list(test_Y_hat)
        assert jso is not None
        assert len(result) == len(PATH) == len(labels)
        for i in range(len(labels)):
            key = labels[i]
            if key not in jso:
                jso[key] = list()
            if result[i] not in jso[key]:
                jso[key].append(result[i])
    return jso


if __name__ == '__main__':
    PATH = librosa.util.find_files('./dataset/valid')
    main(PATH, True)
