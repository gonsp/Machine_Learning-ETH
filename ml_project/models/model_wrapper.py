import numpy as np
from ml_project.pipeline import Pipeline


class ModelWrapper(Pipeline):
    def __init__(self, model, save_path=None):
        super(ModelWrapper, self).__init__(model)

    def fit(self, X, y=None):
        y_new = np.array([np.argmax(p) for p in y])
        super(ModelWrapper, self).fit(X, y_new)
        return self

    def predict_proba(self, X):
        label_order = self._final_estimator.classes_
        y_pred = super(ModelWrapper, self).predict_proba(X)
        y_sorted = np.array([list(zip(*sorted(zip(label_order, p))))[1] for p in y_pred])
        return y_sorted


