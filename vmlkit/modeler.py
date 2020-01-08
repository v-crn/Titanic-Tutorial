from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelFactory(BaseEstimator, ClassifierMixin):
    def __init__(self, name, params):
        self.params = params

        if name == 'LogisticRegression':
            print(params[name])
            self.model = LogisticRegression(**params[name])
        elif name == 'RandomForest':
            self.model = RandomForestClassifier()
        # elif name == 'LightGBM':
        #     self.model = LGBMClassifier()
        else:
            raise NotImplementedError()

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        prediction = self.model.predict_proba(X)
        return prediction

    def predict(self, X, y=None):
        prediction = self.model.predict(X)
        return prediction
