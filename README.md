This project started as part of my bachelor thesis. I continue working on it occasionally.

The goal is to develop an algorithm for sampling the decision boundary of a black-box classifier, whilst keeping a balance between fidelity and speed.

From this near-boundary instance set (NBIS), we can derive a closed interval for each input features such that perturbation within those intervals do not elicit any change on the class predicted by the classifier.

---

You can find a requirements.txt file in the global scope of this repo. However, I warn you, this file was generated using the command:

```python
pip list --format=freeze > requirements.txt
```

So, it is 100% ugly and 0% practical. So much so that it is likely some of the packages are not even necessary!
