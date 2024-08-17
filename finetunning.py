import joblib,librosa, os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import data_loader
from test import main, debug, amountofdata, makecsvofdataset
from train import batch_size, define_model, CreateDataset, Model, timer, runfntg
from train import main as main__

sr = CreateDataset.sr

if makecsvofdataset:
    audios_numpy, labels = data_loader.get_sampels()
    main__(audios_numpy, labels, fit=False)

data_set = pd.read_csv(CreateDataset.Name, index_col=False)
data_set = np.array(data_set)
row, col = data_set.shape  # Cacluate Shape
X_train = data_set[:, :col-1] if not debug else data_set[:3, :col-1]
y_train = data_set[:, col-1] if not debug else np.array(['bass', 'basssssssssss', 'bass', ])

n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

if debug:
    assert X_train.shape[0] == y_train.shape[0] == amountofdata, f'X_train.shape[0]: {X_train.shape[0]}'  # 5501*11
else:
    if not X_train.shape[0] == y_train.shape[0] == amountofdata:
        print(f'X_train.shape[0] should be {amountofdata}, but it is {X_train.shape[0]}')

def finetunning():  # GridSearchCV
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
@runfntg
def run_(clf):  # train all data
    clf.fit(X_train, y_train)
    main(clf=clf)


@timer
@runfntg
def run(clf):  # every_fragment_of_dataset_will_be_trained
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
    main(clf=clf)


if __name__ == '__main__':
    for kw in (
            dict(
                # C=1.0,  # 1.0  # SVC
                n_estimators=None,  # 200  # GradientBoostingClassifier
                criterion=None,  # "entropy",  # DecisionTreeClassifier
                n_estimators_R=None,  # 100,  # RandomForestClassifier
            ),
            dict(
                C=None,  # 1.0  # SVC
                # n_estimators=200,  # 200  # GradientBoostingClassifier
                criterion=None,  # "entropy",  # DecisionTreeClassifier
                n_estimators_R=None,  # 100,  # RandomForestClassifier
            ),
            dict(
                C=None,  # 1.0  # SVC
                n_estimators=None,  # 200  # GradientBoostingClassifier
                # criterion="entropy",  # "entropy",  # DecisionTreeClassifier
                n_estimators_R=None,  # 100,  # RandomForestClassifier
            ),
            dict(
                C=None,  # 1.0  # SVC
                n_estimators=None,  # 200  # GradientBoostingClassifier
                criterion=None,  # "entropy",  # DecisionTreeClassifier
                # n_estimators_R=100,  # 100,  # RandomForestClassifier
            ),
    ):
        run_(**kw)
