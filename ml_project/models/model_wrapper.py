import numpy as np
from ml_project.pipeline import Pipeline


class ModelWrapper(Pipeline):
    def __init__(self, model, save_path=None):
        self.model = model
        super(ModelWrapper, self).__init__(self.model)

    def fit(self, X, y=None):
        y_new = np.array([np.argmax(p) for p in y])
        super(ModelWrapper, self).fit(X, y_new)
        return self

    def transform(self, X, y=None):
        print("This shouldn't happen")
        exit(1)
        return X
