import joblib, os, sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from feature_engineering import *
from config import *

PATH = librosa.util.find_files(Test_path.data_path)


def Evaluate_model(y_true, y_pred):
    # site: https://machinelearningcoban.com/2017/08/31/evaluation/
    labels = np.array(y_true)
    print('test accuracy = ', accuracy_score(labels, y_pred), ' %')

    print(classification_report(labels, y_pred))

    cnf_matrix = confusion_matrix(labels, y_pred)
    print('Confusion matrix:\n', cnf_matrix)
    print('\nAccuracy:', np.diagonal(cnf_matrix).sum() / cnf_matrix.sum())


def main():
    labels = []
    samples = []
    for p in PATH:
        labels.append(p.split(os.sep)[-1].split('_')[0])
        sample, sr = librosa.load(p, sr=22050, duration=4.0)
        samples.append(sample)

    data = np.array([extract_feature(sample) for sample in tqdm(samples)])

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data)

    clf = joblib.load(Model.NAME)
    test_Y_hat = clf.predict(data)

    accuracy = np.sum((test_Y_hat == labels)) / 200.0 * 100.0
    Evaluate_model(labels, test_Y_hat)

    print('test accuracy = ' + str(accuracy) + ' %')

if __name__ == '__main__':
    main()
