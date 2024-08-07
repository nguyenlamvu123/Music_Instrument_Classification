import data_loader
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import librosa, joblib, os

from train import readdata, batch_size, define_model, CreateDataset, Model
from sklearn.metrics import classification_report

sr = CreateDataset.sr

data_set = pd.read_csv(CreateDataset.Name, index_col=False)
data_set = np.array(data_set)
row, col = data_set.shape  # Cacluate Shape
X_train = data_set[:, :col-1]
y_train = data_set[:, col-1]

# assert X_train.shape[0] == y_train.shape[0] == 289205

n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': [
        'rbf',
        # 'linear'
    ]
}

X_test, y_test = data_loader.get_sampels('test')
assert len(X_test) == len(y_test) == 4096
samples = readdata(X_test)  # for audio in tqdm(X_test):
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
for C in param_grid['C'][::-1]:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            modelname = f'{os.path.splitext(Model.NAME)[0]}' + \
                         f'__C_{C}__gamma_{gamma}__kernel_{kernel}{os.path.splitext(Model.NAME)[-1]}'
            grid = SVC(
                C=C,
                cache_size=200,
                class_weight=None,
                coef0=0.0,
                decision_function_shape='ovr',
                degree=3,
                gamma=gamma,
                kernel=kernel,
                max_iter=-1,
                probability=False,
                random_state=None,
                shrinking=True,
                tol=0.001,
                verbose=False
            )
            print(f'################### {modelname}')

            grid.fit(X_train, y_train)
            joblib.dump(grid, modelname)

            grid_predictions = grid.predict(samples)
            print(classification_report(y_test, grid_predictions))
