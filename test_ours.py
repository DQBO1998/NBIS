import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
import os
import argparse
import pandas as pd
from itertools import product
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.datasets import make_moons
from nbis import nbis, fit_aabb, dilate
from tqdm import tqdm
from colorama import Fore
from pathos.multiprocessing import Pool
from time import time


class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})


def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./'):

    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'diabetes':
        categorical_features = [2, 3, 4, 5, 6, 7, 8, 10, 11, 18, 19, 20, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                47, 48]
        label_map = {'<30': 'YES', '>30': 'YES'}
        transformations = {
            49: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes/diabetic_data.csv'), -1, ',',
            features_to_use=range(2, 49),
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'default':
        categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11]
        dataset = load_csv_dataset(
                os.path.join(dataset_folder, 'default/default.csv'), -1, ',',
                features_to_use=range(1, 24),
                categorical_features=categorical_features, discretize=discretize,
                balance=balance)
    elif dataset_name == 'recidivism':
        features_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                         'Married', 'Felony', 'WorkRelease',
                         'Crime against Property', 'Crime against Person',
                         'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                         'Age', 'MonthsServed', '', 'Recidivism']
        def violations_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})
        def priors_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})
        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: priors_fn(x),
            12: lambda x: violations_fn(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes',
                                                    'Re-arrested'])
        }

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            features_to_use=features_to_use, balance=balance,
            feature_transformations=transformations, skip_first=True)
    elif dataset_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
               'Does not meet the credit policy. Status:Fully Paid',
               'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data
        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16:  lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19:  lambda x: np.array([len(y) for y in x]).astype(int),
            6:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_features = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'),
            16, ',',  features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_features=categorical_features, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        dataset.class_names = ['Good Loan', 'Bad Loan']
    elif dataset_name == 'moons':
        data, labels = make_moons(n_samples=1000, noise=0.1)
        ret = Bunch({})
        ret.labels = labels
        splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                      test_size=.2,
                                                      random_state=1)
        train_idx, test_idx = [x for x in splits.split(data)][0]
        ret.train = data[train_idx]
        ret.labels_train = ret.labels[train_idx]
        cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                        test_size=.5,
                                                        random_state=1)
        cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
        cv_idx = test_idx[cv_idx]
        test_idx = test_idx[ntest_idx]

        ret.validation = data[cv_idx]
        ret.labels_validation = ret.labels[cv_idx]
        ret.test = data[test_idx]
        ret.labels_test = ret.labels[test_idx]
        ret.test_idx = test_idx
        ret.validation_idx = cv_idx
        ret.train_idx = train_idx
        ret.data = data
        dataset = ret
    return dataset


def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = list(range(data.shape[1]))
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    ret.data = data
    return ret


def coverage(aabb: tuple[np.ndarray, np.ndarray], X: np.ndarray):
    neg, pos = aabb
    X_in = X[np.all(neg <= X, axis=1) & np.all(X <= pos, axis=1)]
    return X_in.shape[0] / X.shape[0]


def precision(aabb: tuple[np.ndarray, np.ndarray], X: np.ndarray, y: np.ndarray, ref: np.ndarray):
    neg, pos = aabb
    y_in = y[np.all(neg <= X, axis=1) & np.all(X <= pos, axis=1)]
    return y_in[y_in == ref].shape[0] / (y_in.shape[0] + 1e-5)


def make_expl(model, data, 
              nbis_n_samples: int | None = 500, nbis_n_trials: int = 2, nbis_max_iter: int = 100, 
              expl_n_samples: tuple = (100, 100), expl_n_trials: int = 5, expl_n_jobs: int = 1):
    nbis_n_samples = np.inf if nbis_n_samples is None else nbis_n_samples
    def _make_nbis(idx: int):
        X0 = data.train[np.random.rand(data.train.shape[0]) <= nbis_n_samples / data.train.shape[0]]
        scale = 3.
        tf = make_pipeline(StandardScaler(), 
                           FunctionTransformer(func=lambda X: X * scale, 
                                               inverse_func=lambda X: X / scale)).fit(X0)
        for st in tqdm(nbis(X0=tf.transform(X0), 
                            func=lambda X: model.predict_proba(tf.inverse_transform(X)), 
                            outer_steps=nbis_max_iter, 
                            noise=0.05),
                       total=nbis_max_iter,
                       desc=f'nbis {idx}',
                       disable=False):
            pass
        return tf.inverse_transform(st.X)
    Xs = [_make_nbis(i + 1) for i in range(nbis_n_trials)]
    Xt = np.concatenate(Xs, axis=0)
    def _make_expl():
        X0 = data.train[np.random.rand(data.train.shape[0]) <= expl_n_samples[0] / (2 * data.train.shape[0])]
        _Xt = Xt[np.random.rand(Xt.shape[0]) <= expl_n_samples[1] / (2 * Xt.shape[0])]
        def _expl(x: np.ndarray):
            return fit_aabb(x=x, 
                            supp=X0, 
                            db=_Xt, 
                            func=model.predict, 
                            verbose=False,
                            M=1000000.)
        return _expl
    expls = [_make_expl() for _ in range(expl_n_trials)]
    def _ensemble_expl(x: np.ndarray):
        if expl_n_jobs == 1:
            res = [expl(x) for expl in expls]
        else:
            with Pool(expl_n_jobs) as pool:
                res = pool.map(func=lambda fn: fn(x), iterable=expls)
        aabbs = [(neg, pos) for status, neg, pos in res if status]
        if aabbs:
            neg, pos = zip(*aabbs)
            neg = np.stack(neg, axis=0); pos = np.stack(pos, axis=0)
            return True, np.median(neg, axis=0), np.median(pos, axis=0)
        return False, None, None
    return _ensemble_expl


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-d', '--dataset', type=str, dest='dataset_name', 
                      choices=['moons', 'adult', 'lending', 'recidivism'], default='moons')
    argp.add_argument('-m', '--model', type=str, dest='model_name', choices=['gb', 'lr', 'nn', 'dt'], default='dt')
    argp.add_argument('-i', '--input', type=str, dest='input_folder', default='D:/Github/NBIS/datasets/')
    argp.add_argument('-o', '--output', type=str, dest='output_folder', default='D:/Github/NBIS/our_results/')
    argp.add_argument('-t', '--take', type=int, dest='take', default=10000)
    args = argp.parse_args()

    data = load_dataset(dataset_name=args.dataset_name, discretize=False, dataset_folder=args.input_folder)
    models = {'gb': GradientBoostingClassifier(),
              'dt': DecisionTreeClassifier(),
              'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=700)),
              'nn': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(50, 50), 
                                                                  max_iter=5000, 
                                                                  early_stopping=True, 
                                                                  tol=0.0001))}
    
    print(f'DATASET: {Fore.LIGHTCYAN_EX}{args.dataset_name}{Fore.RESET}')
    print(f'\tFEATURES: {Fore.LIGHTGREEN_EX}{data.train.shape[1]}{Fore.RESET}')
    n_classes = np.unique(data.labels_train).shape[0]
    print(f'\tCLASSES: {Fore.LIGHTGREEN_EX}{n_classes}{Fore.RESET}')
    print(f'\tTRAIN: {Fore.LIGHTGREEN_EX}{data.train.shape[0]}{Fore.RESET}')
    print(f'\tTEST: {Fore.LIGHTGREEN_EX}{data.test.shape[0]}{Fore.RESET}')
    print(f'\tVALIDATION: {Fore.LIGHTGREEN_EX}{data.validation.shape[0]}{Fore.RESET}')

    print(f'MODEL: {Fore.LIGHTRED_EX + args.model_name + Fore.RESET}')
    model = copy.deepcopy(models[args.model_name]).fit(data.train, data.labels_train)
    train_acc = model.score(data.train, data.labels_train)
    print(f'\tTRAIN ACCURACY: {Fore.LIGHTGREEN_EX}{train_acc}{Fore.RESET}')
    _X = np.concatenate([data.validation, data.test], axis=0)
    _y = np.concatenate([data.labels_validation, data.labels_test], axis=0)
    test_acc = model.score(_X, _y)
    print(f'\tTEST ACCURACY: {Fore.LIGHTGREEN_EX}{test_acc}{Fore.RESET}')

    expl = make_expl(model=model, data=data)

    _X = data.validation[:args.take]
    results = [('model', 'dataset', 'train_acc', 'test_acc', 'coverage', 'precision', 'missed', 'time')]
    for x in tqdm(_X,
                  desc=f'expl',
                  total=_X.shape[0]):
        t1 = time()
        flag, neg, pos = expl(x)
        t2 = time()
        aabb = (neg, pos)
        if flag:
            cov = coverage(aabb=aabb, X=data.test)
            prec = precision(aabb=aabb, X=data.test, y=model.predict(data.test), ref=model.predict(np.atleast_2d(x))[0])
            miss = False
        else:
            cov = 0.
            prec = 0.
            miss = True
        span = t2 - t1
        results.append((args.model_name, args.dataset_name, train_acc, test_acc, cov, prec, miss, span))

    df = pd.DataFrame(data=results[1:], columns=results[0])
    print(df[['coverage', 'precision', 'missed', 'time']].mean(axis=0))
    df.to_csv(f'{args.output_folder}{args.model_name}_{args.dataset_name}.csv', index=False)

