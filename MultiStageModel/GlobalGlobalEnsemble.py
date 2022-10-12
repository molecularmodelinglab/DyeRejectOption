import numpy as np
from copy import deepcopy


class GlobalGlobalEnsemble:
    def __init__(self, classifier, regressor, bins_list):
        self.c = classifier
        self.r = regressor
        self.cs = []
        self.bins_list = bins_list

    def __name__(self):
        return f"{self.__class__.__name__}_{self.c.__class__.__name__}_{self.r.__class__.__name__}_{'_'.join([str(len(_)+1) for _ in self.bins_list])}"

    def fit(self, X, y):
        for bins in self.bins_list:
            classes = np.digitize(y, bins, right=True)
            c_copy = deepcopy(self.c)
            c_copy.fit(X, classes)
            self.cs.append(c_copy)
        self.r.fit(X, y)

    def predict(self, X, reduce=None):
        return self.r.predict(X)

    def predict_proba(self, X, reduce="mean"):
        confidence = []
        for c in self.cs:
            conf = np.max(np.array(c.predict_proba(X)), axis=-1)
            confidence.append(conf)
        if reduce == "mean":
            confidence = np.mean(np.array(confidence).T, axis=-1)
        elif reduce == "max":
            confidence = np.max(np.array(confidence).T, axis=-1)
        return confidence
