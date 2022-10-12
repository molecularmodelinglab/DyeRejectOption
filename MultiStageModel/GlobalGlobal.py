import numpy as np


class GlobalGlobal:
    def __init__(self, classifier, regressor, classifier_pretrained=False, regressor_pretrained=False,
                 bins=None):
        self.c = classifier
        self.r = regressor

        self.train_classifier = not classifier_pretrained
        self.train_reg = not regressor_pretrained

        self._bin_name = str(len(bins)+1) if bins is not None else "color"
        self.bins = bins if bins is not None else [380, 450, 495, 570, 590, 620, 750]

    def __name__(self):
        return f"{self.__class__.__name__}_{self.c.__class__.__name__}_{self.r.__class__.__name__}_{self._bin_name}"

    def fit(self, X, y):
        bins = np.digitize(y, self.bins, right=True)

        if self.train_classifier:
            self.c.fit(X, bins)

        if self.train_reg:
            self.r.fit(X, y)

    def predict(self, X):
        return self.r.predict(X)

    def predict_proba(self, X):
        bins = self.c.predict_proba(X)
        return np.array(bins)

