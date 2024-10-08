import data_loader
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import librosa, joblib

from train import readdata, batch_size, define_model, CreateDataset, Model, timer
from test import main
from sklearn.metrics import classification_report, confusion_matrix

sr = CreateDataset.sr

data_set = pd.read_csv(CreateDataset.Name, index_col=False)
data_set = np.array(data_set)
row, col = data_set.shape  # Cacluate Shape
X_train = data_set[:, :col-1]
y_train = data_set[:, col-1]

n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

assert X_train.shape[0] == y_train.shape[0] == 289204, f'X_train.shape[0]: {X_train.shape[0]}'  # 289205

def finetunning():
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print('@@@@@@@', grid.best_estimator_)

    X_test, y_test = data_loader.get_sampels('test')
    assert len(X_test) == len(y_test) == 4096
    samples = []
    for p in X_test:
        sample, _ = librosa.load(p, sr=sr, duration=4.0)
        samples.append(sample)

    grid_predictions = grid.predict(samples)
    print(classification_report(y_test, grid_predictions))


@timer
def run_():
    clf = GradientBoostingClassifier(n_estimators=200, random_state=0, max_depth=2)
    clf.fit(X_train, y_train)
    main(clf=clf)


@timer
def run(C=1.0, gamma=0.02, degree=3, coef0=0.0, ):
    # clf = define_model(C=C, gamma=gamma, degree=degree, coef0=coef0)  # SVC
    # clf = GradientBoostingClassifier(n_estimators=200, random_state=0, max_depth=2)  # Accuracy: 0.202880859375
    # clf = DecisionTreeClassifier(criterion="entropy")  # Accuracy: 0.168701171875
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf.fit(X_train, y_train)
    for i in range(n_batches):
        print(f"{i + 1}/{n_batches}")
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_y = y_train[start_idx:end_idx]
        assert len(set(batch_y)) > 1
        batch_X = X_train[start_idx:end_idx]
        clf.fit(batch_X, batch_y)

    if n_samples % batch_size != 0:
        remaining_y = y_train[n_batches * batch_size:]
        if len(set(remaining_y)) > 1:
            remaining_X = X_train[n_batches * batch_size:]
            clf.fit(remaining_X, remaining_y)
    joblib.dump(clf, f'C_{C}__gamma_{gamma}__' + Model.NAME)
    main(clf=clf)


if __name__ == '__main__':
    # run(C=10, gamma=0.1, )  # 0.262939453125
    # run(C=20, gamma=0.1, )  #
    # run(C=1000, gamma=1, )  #
    run()  #
