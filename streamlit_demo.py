import streamlit as st
import sys
from nbis import nbis
from nbis import fit_aabb
from stqdm import stqdm
from typing import Sequence, Any, Callable
from test_ours import precision, coverage


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


SCALE = 3.


def make_sgnxor(n_samples: int = 100, noise: float = 0.5):
    X = np.random.randn(n_samples, 2)
    y = X.prod(axis=1) >= 0.
    X = X + np.random.randn(n_samples, 2) * noise
    return X, y


def make_spirals(n_samples: int = 100, noise=0.2):
    n_points = n_samples
    n = np.sqrt(np.random.rand(n_points, 1)) * 720 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y


def make_circle(n_samples=100, radius=1.0, noise=0.1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.uniform(0, radius * 1.5, n_samples)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    X += np.random.normal(scale=noise, size=X.shape)
    y = radii < radius
    return X, y.astype(int)


def plot_dbs(classifiers, X_list, y_list, steps=1000, cmaps=['viridis'], titles=None, neg=None, pos=None, loc=None, scale: float = 1.):
    n_classifiers = len(classifiers)
    fig, axs = plt.subplots(1, n_classifiers, figsize=(15, 5 * n_classifiers))
    x_min, x_max = -2. * scale, 2. * scale
    y_min, y_max = -2. * scale, 2. * scale
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    for i, classifier in enumerate(classifiers):
        ax = axs[i] if n_classifiers > 1 else axs
        X, y = X_list[i], y_list[i]
        Z = classifier(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        cmap = cmaps[i] if i < len(cmaps) else cmaps[0]
        ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cmap, extent=(xx.min(), xx.max(), yy.min(), yy.max()))
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap, marker='.')
        if neg is not None and pos is not None:
            neg[0] = max(neg[0], xx.min())
            neg[1] = max(neg[1], yy.min())
            pos[0] = min(pos[0], xx.max())
            pos[1] = min(pos[1], yy.max())
            rect = plt.Rectangle((neg[0], neg[1]), 
                                pos[0] - neg[0], 
                                pos[1] - neg[1], 
                                linewidth=1, 
                                edgecolor='white', 
                                facecolor='none')
            ax.add_patch(rect)
        if loc is not None:
            ax.scatter(*loc, color='white', marker='o')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        title = titles[i] if titles and i < len(titles) else f'Classifier {i + 1}'
        ax.set_title(title)
    fig.tight_layout()
    return fig


datasets = {'Moons': (make_moons, {'noise': 0.1}), 
            'Circle': (make_circle, {'noise': 0.1}), 
            'Clusters': (make_classification, {'n_classes': 2, 'n_features': 2, 'n_informative': 2, 'n_redundant': 0}),
            'Spirals': (make_spirals, {'noise': 0.1}),
            'Signed XOR': (make_sgnxor, {'noise': 0.3})}
models = {'K-Nearest-Neighbors Classifier': KNeighborsClassifier, 
          'Decision Tree Classifier': DecisionTreeClassifier,
          'Random Forest Classifier': RandomForestClassifier,
          'Gaussian Process Classifier': GaussianProcessClassifier,
          'Gaussian Naive Bayes': GaussianNB}


if 'global_count' not in st.session_state:
    st.session_state.global_count = 1
local_count = st.session_state.global_count


def _countup(*args, **kwargs):
    st.session_state.global_count += 1


@st.cache_data
def make_dataset(dataset: str, n_samples: int):
    method, def_params = datasets[dataset]
    X, y = method(n_samples=n_samples, **def_params)
    return StandardScaler().fit_transform(X) * SCALE, y


@st.cache_data
def model_and_data(model_id: str, dataset_id: str, n_samples: int):
    X0, y0 = make_dataset(dataset=dataset_id, n_samples=n_samples)
    model = models[model_id]().fit(X0, y0)
    return model, X0, y0


@st.cache_data
def make_nbis(X0: np.ndarray, model_id: str, max_iter: int, noise: float):
    nbis_iter = enumerate(nbis(X0=X0, 
                               func=st.session_state.model.predict_proba, 
                               outer_steps=max_iter,
                               noise=noise))
    for i, res in stqdm(nbis_iter, total=max_iter):
        if local_count < st.session_state.global_count:
            sys.exit()
    return res.X


@st.cache_data
def make_aabb(X0: np.ndarray, Xt: np.ndarray, model_id: str, x1: float, x2: float):
    loc = np.array([x1, x2])
    neg = None
    pos = None
    _, neg, pos = fit_aabb(x=loc, 
                           supp=X0, 
                           db=Xt, 
                           func=st.session_state.model.predict, 
                           verbose=True)
    return loc, neg, pos


with st.sidebar:
    st.radio(label='Dataset', options=datasets.keys(), key='dataset_id', index=0, horizontal=True, on_change=_countup)
    st.slider(label='Sample size', min_value=50, max_value=1000, value=100, step=10, key='n_samples', on_change=_countup)
    st.radio(label='Model', options=models.keys(), key='model_id', index=0, horizontal=True, on_change=_countup)
    '---'
    st.radio(label='Overlay', options=['Original', 'NBIS'], key='overlay', index=0, horizontal=True, on_change=_countup)
    st.slider(label='$x_1$', min_value=-2. * SCALE, max_value=2. * SCALE, value=0., key='x1', on_change=_countup)
    st.slider(label='$x_2$', min_value=-2. * SCALE, max_value=2. * SCALE, value=0., key='x2', on_change=_countup)
    '---'
    st.slider(label='Max iterations', min_value=1, max_value=1000, value=50, step=10, key='max_iter', on_change=_countup)
    st.slider(label='Noise level', min_value=0.001, max_value=1., value=0.1, step=0.001, key='noise_lvl', on_change=_countup)

model, X0, y0 = model_and_data(model_id=st.session_state.model_id, 
                               dataset_id=st.session_state.dataset_id, 
                               n_samples=st.session_state.n_samples)
st.session_state.model = model

f'''
### Inferring Local Rules from Near Boundary Instances: 2D Demo

For more information, visit [NBIS on GitHub](https://github.com/DQBO1998/NBIS).

---
'''

Xt = make_nbis(X0=X0, model_id=st.session_state.model_id, max_iter=st.session_state.max_iter, noise=st.session_state.noise_lvl)
loc, neg, pos = make_aabb(X0=X0, Xt=Xt, model_id=st.session_state.model_id, x1=st.session_state.x1, x2=st.session_state.x2)

f'''
C: {coverage(aabb=(neg, pos), X=X0)} | P: {precision(aabb=(neg, pos), X=X0, y=y0, ref=model.predict(np.atleast_2d(loc))[0])}
'''

fig = plot_dbs(classifiers=[model.predict_proba], 
               X_list=[X0 if st.session_state.overlay == 'Original' else Xt],
               y_list=[model.predict(X0 if st.session_state.overlay == 'Original' else Xt)],
               titles=[st.session_state.overlay],
               steps=100,
               cmaps=['viridis'],
               neg=neg,
               pos=pos,
               loc=loc,
               scale=SCALE)
st.pyplot(fig, clear_figure=True)
plt.close(fig)