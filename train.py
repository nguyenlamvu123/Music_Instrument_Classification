import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
import joblib, os
from functools import wraps

from config import Model, CreateDataset, debug, batch_size
from feature_engineering import readdata, timer

csv_name = CreateDataset.Name


def define_model(
        C: float or None = None,
        n_estimators: int or None = None,
        criterion: str or None = None,
        n_estimators_R: int or None = None,
        **kwargs,
):
    if C is not None:  # SVC
        assert all([s is None for s in (n_estimators, criterion, n_estimators_R)])
        assert all([s in kwargs for s in ('gamma', 'degree', 'coef0', )])
        gamma, degree, coef0 = kwargs['gamma'], kwargs['degree'], kwargs['coef0']
        print(f'SVC__C={C}__gamma={gamma}__degree={degree}__coef0={coef0}')
        model = svm.SVC(
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
    elif n_estimators is not None:  # GradientBoostingClassifier
        assert all([s is None for s in (C, criterion, n_estimators_R)])
        assert all([s in kwargs for s in ('random_state', 'max_depth', )])
        random_state, max_depth = kwargs['random_state'], kwargs['max_depth']
        print(f'GradientBoostingClassifier__n_estimators={n_estimators}__random_state={random_state}__max_depth={max_depth}')
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth
        )
    elif criterion is not None:  # DecisionTreeClassifier
        assert all([s is None for s in (C, n_estimators, n_estimators_R)])
        print(f'DecisionTreeClassifier__criterion={criterion}')
        model = DecisionTreeClassifier(criterion=criterion)
    elif n_estimators_R is not None:  # RandomForestClassifier
        assert all([s is None for s in (C, criterion, n_estimators)])
        assert all([s in kwargs for s in ('random_state', 'max_depth', )])
        random_state, max_depth = kwargs['random_state'], kwargs['max_depth']
        print(f'RandomForestClassifier__n_estimators={n_estimators_R}__random_state={random_state}__max_depth={max_depth}')
        model = RandomForestClassifier(
            n_estimators=n_estimators_R,
            random_state=random_state,
            max_depth=max_depth
        )
    else:
        raise
    return model


def runfntg(func):  # @runfntg
    @wraps(func)
    def wrapper(
            C: float or None = 1.0, gamma=0.02, degree=3, coef0=0.0,  # SVC
            n_estimators: int or None = 200, random_state=0, max_depth=50,  # GradientBoostingClassifier
            criterion: str or None = "entropy",  # DecisionTreeClassifier
            n_estimators_R: int or None = 100,  # RandomForestClassifier
    ):
        modelname = f'{os.path.splitext(Model.NAME)[0]}__'
        if C is not None:
            kw = dict(C=C, gamma=gamma, degree=degree, coef0=coef0)                                                     # SVC
            modelname += f'SVC__{C}__{gamma}{os.path.splitext(Model.NAME)[-1]}'
        elif n_estimators is not None:
            kw = dict(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)                        # GradientBoostingClassifier
            modelname += f'GradientBoostingClassifier__{n_estimators}__{random_state}{os.path.splitext(Model.NAME)[-1]}'
        elif criterion is not None:
            kw = dict(criterion=criterion)                                                                              # DecisionTreeClassifier
            modelname += f'DecisionTreeClassifier__{criterion}{os.path.splitext(Model.NAME)[-1]}'
        elif n_estimators_R is not None:
            kw = dict(n_estimators_R=n_estimators_R, random_state=random_state, max_depth=max_depth, warm_start=True)   # RandomForestClassifier
            modelname += f'RandomForestClassifier__{n_estimators_R}__{random_state}{os.path.splitext(Model.NAME)[-1]}'
        else:
            raise
        clf = define_model(**kw)
        result = func(clf)
        joblib.dump(clf, modelname)
        return result
    return wrapper

def main(audios_numpy, labels, fit: bool = True):
    # Load data
    n_samples = audios_numpy.shape[0]
    n_batches = n_samples // batch_size

    clf = define_model() if fit else None

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

        if not fit:
            continue
        assert clf is not None
        clf.fit(batch_X, batch_y)
        joblib.dump(clf, Model.NAME)

    if n_samples % batch_size != 0:
        piece = audios_numpy[n_batches * batch_size:]
        remaining_y = labels[n_batches * batch_size:]
        if len(set(remaining_y)) > 1:
            remaining_X = readdata(piece)  # for audio in tqdm(piece):

            dataset_pandas = pd.DataFrame(remaining_X)
            dataset_pandas["instruments"] = remaining_y
            dataset_pandas.to_csv(csv_name, mode='a', index=False, header=False)

            if not fit:
                return
            assert clf is not None
            clf.fit(remaining_X, remaining_y)
            joblib.dump(clf, Model.NAME)
