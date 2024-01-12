import streamlit as st
from demo_utils import *
from typing import Callable
from nbis import nbis
from stqdm import stqdm
from nbis import refine

with st.sidebar:
    st.radio(label='Dataset', options=datasets.keys(), key='dataset_id', index=0, horizontal=True)
    st.radio(label='Model', options=models.keys(), key='model_id', index=0, horizontal=False)
    st.checkbox(label='Retain $X_0$ while training', value=True, key='keep_X0')
    st.checkbox(label='Re-sample while training', value=True, key='sample_aabb')
    st.slider(label='Sample size', min_value=50, max_value=1000, value=100, key='n_samples')
    st.slider(label='Max iterations', min_value=1, max_value=1000, value=50, key='max_iter')
    st.slider(label='Noise level', min_value=0.0001, max_value=1., value=0.06, key='noise_lvl')

X, y = make_dataset(dataset=st.session_state.dataset_id, n_samples=st.session_state.n_samples)
Y = np.zeros((X.shape[0], 2))
Y[y == 0, 0] = 1
Y[y == 1, 1] = 1
model = models[st.session_state.model_id]().fit(X, y)
classifier = lambda X: model.predict_proba(X)

f'''
# Near-Boundary Instance Sets (NBIS): Demo

A near-boundary instance set (NBIS) is defined as a set of examples scattered very close to the decision boundary of a black-box classifier. This interactive demo helps readers become familiar with the process of generating NBISs in 2-dimensional domains.

**Note:** While it is perfectly possible to generate NBISs in higher dimensions, it is not possible to visualize them reliably. So, this demo works exclusively with 2-dimensional datasets.

The following figure shows the decision boundary of a {st.session_state.model_id} trained on {st.session_state.dataset_id} dataset. You may use the controllers on the left to switch to a different model or dataset.
'''

st.pyplot(plot_dbs([classifier], 
                   X, 
                   Y=Y, 
                   titles=['Target model'], 
                   steps=100, 
                   cmaps=['viridis']), clear_figure=True)

f'''
The following figure shows the obtained NBIS, the decision boundary of the black-box classifier, and its approximation. You may change the number of steps by moving the slider labeled "Max iterations". Right now it is set to {st.session_state.max_iter}. This parameter controls for how long the algorithm refines the NBIS.
'''

for i, res in stqdm(enumerate(nbis(X0=X, 
                                   func=classifier, 
                                   outer_steps=st.session_state.max_iter,
                                   keep_X0=st.session_state.keep_X0,
                                   sample_aabb=st.session_state.sample_aabb,
                                   noise=st.session_state.noise_lvl)), total=st.session_state.max_iter):
    pass
st.pyplot(plot_dbs(classifiers=[classifier, res.predict_proba], 
                   X=res.X,
                   Y=Y,
                   titles=['NBIS overlay with target model', 'NBIS overlay with surrogate model'],
                   steps=100,
                   cmaps=['viridis', 'viridis']), clear_figure=True)

'''
---
I developed this algorithm for my bachelor's thesis. If you find the topic interesting or you are simply curious about the project, I would like to encourage you to visit [NBIS on GitHub](https://github.com/DQBO1998/NBIS). That is the link to the project repository.
'''
