import numpy as np
from copy import deepcopy


class GlobalGlobalEnsemble:
    def __init__(self, classifier, regressor, bins_list=None, num_ensemble=4, num_cutoffs=11):
        self.c = classifier
        self.r = regressor
        self.cs = []
        self.bins_list = bins_list
        self.num_ensemble = num_ensemble
        self.num_cutoffs = num_cutoffs

    def __name__(self):
        return f"{self.__class__.__name__}_{self.c.__class__.__name__}_{self.r.__class__.__name__}_{'_'.join([str(len(_)+1) for _ in self.bins_list])}"

    def make_bins(self, y, num_cutoffs=None, num_sets=None):
        if num_cutoffs is None:
            num_cutoffs = self.num_cutoffs
        if num_sets is None:
            num_sets = self.num_ensemble

        bottom_5 = np.quantile(y, 1 / 20)
        top_5 = np.quantile(y, 19 / 20)
        bin_size = np.abs(top_5 - bottom_5) / num_cutoffs

        if num_sets == 1:
            self.bins_list = [bottom_5 + (bin_size * x) for x in range(num_cutoffs)]
        else:
            bins = [[bottom_5 + (bin_size * x) for x in range(num_cutoffs)]]
            bin_step = bin_size / num_sets
            for _ in range(1, num_sets):
                bins.append([bottom_5 + (bin_size * x) + (bin_step * _) for x in range(num_cutoffs)])
            self.bins_list = bins

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
