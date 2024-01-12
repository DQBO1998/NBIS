import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB


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


def plot_dbs(classifiers, X, Y=None, steps=1000, cmaps=['viridis'], titles=None, path: bool = False):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))

    n_classifiers = len(classifiers)
    fig, axs = plt.subplots(1, n_classifiers, figsize=(15, 5 * n_classifiers))

    for i, classifier in enumerate(classifiers):
        ax = axs[i] if n_classifiers > 1 else axs
        Z = classifier(np.c_[xx.ravel(), yy.ravel()])[:, 0].reshape(xx.shape)
        cmap = cmaps[i] if i < len(cmaps) else cmaps[0]
        ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cmap, extent=(xx.min(), xx.max(), yy.min(), yy.max()))
        if path:
            ax.plot(X[:, 0], X[:, 1], '--', c='black')
        else:
            ax.scatter(X[:, 0], X[:, 1], c=Y[:, 0], edgecolors='k', cmap=cmap, marker='o')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        title = titles[i] if titles and i < len(titles) else f'Classifier {i + 1}'
        ax.set_title(title)

    fig.tight_layout()
    return fig


def make_dataset(dataset: str, n_samples: int):
    method, def_params = datasets[dataset]
    return method(n_samples=n_samples, **def_params)


datasets = {'Moons': (make_moons, {'noise': 0.1}), 
            'Circle': (make_circle, {'noise': 0.1}), 
            'Clusters': (make_classification, {'n_classes': 2, 'n_features': 2, 'n_informative': 2, 'n_redundant': 0}),
            'Spirals': (make_spirals, {'noise': 0.1})}
models = {'K-Nearest-Neighbors Classifier': KNeighborsClassifier, 
          'Decision Tree Classifier': DecisionTreeClassifier,
          'Gradient Boosting Classifier': GradientBoostingClassifier,
          'Gaussian Process Classifier': GaussianProcessClassifier,
          'Gaussian Naive Bayes': GaussianNB}
