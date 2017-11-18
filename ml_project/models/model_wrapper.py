import numpy as np
from ml_project.pipeline import Pipeline


class ModelWrapper(Pipeline):
    def __init__(self, model, save_path=None):
        super(ModelWrapper, self).__init__(model)

    def fit(self, X, y=None):
        y_new = np.array(["".join([str(i) for i in np.argsort(-p)]) for p in y])
        super(ModelWrapper, self).fit(X, y_new)
        return self

    def predict_proba(self, X):
        labels = self._final_estimator.classes_
        y_pred = super(ModelWrapper, self).predict_proba(X)
        y_sorted = np.array([self.label_to_probability(max(list(zip(labels, p)), key=lambda k: k[1])[0]) for p in y_pred])
        return y_sorted

    def label_to_probability(self, label):
        l = list(zip([0.4, 0.3, 0.2, 0.1], label))
        l = sorted(l, key=lambda k: k[1])
        return list(zip(*l))[0]