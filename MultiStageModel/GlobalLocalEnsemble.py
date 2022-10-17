import numpy as np
from copy import deepcopy


class GlobalLocalEnsemble:
    def __init__(self, classifier, regressor, bins_list=None, num_ensemble=4, num_cutoffs=11):
        self.c = classifier
        self.r = regressor

        self.models = []

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

            regs = []
            for _ in range(len(bins) + 1):
                r_c = deepcopy(self.r)
                X_bin = X[classes == _]
                y_bin = y[classes == _]
                r_c.fit(X_bin, y_bin)
                regs.append(r_c)
            self.models.append((c_copy, regs))

    def predict(self, X, reduce="mean", with_confidence_weights=True):
        preds_overall = []
        for c, r in self.models:
            bins = c.predict(X)
            preds = []
            for i, row in enumerate(X):
                preds.append(r[bins[i]].predict(row.reshape(1, -1)))
            preds_overall.append(np.array(preds))
        preds_overall = np.array(preds_overall).squeeze(axis=-1).T

        if reduce == "mean":
            if with_confidence_weights:
                from scipy.special import softmax
                confidence = self.predict_proba(X, reduce="none")
                confidence = softmax(np.array(confidence).T, axis=-1)
                return np.sum(confidence * preds_overall, axis=-1)
            else:
                return np.mean(preds_overall, axis=-1)
        elif reduce == "max":
            conf_idx = self.predict_proba(X, reduce="max", _arg_max=True)
            return np.choose(conf_idx, preds_overall.T)

    def predict_proba(self, X, reduce="mean", _arg_max=False):
        confidence = []
        for c, r in self.models:
            conf = np.max(np.array(c.predict_proba(X)), axis=-1)
            confidence.append(conf)
        if reduce == "mean":
            confidence = np.mean(np.array(confidence).T, axis=-1)
        elif reduce == "max":
            if _arg_max:
                return np.argmax(np.array(confidence).T, axis=-1)
            confidence = np.max(np.array(confidence).T, axis=-1)
        elif reduce == "none":
            confidence = np.array(confidence)
        return confidence
