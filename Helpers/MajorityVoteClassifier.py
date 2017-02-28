
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A Majority Vote Classifier"""

    def __init__(self, classifiers, vote='classlabel', weights=None):
        """ Init """
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers"""
        # Use LabelEncoder to encode target classes
        # from 0 to n
        self.lablenc_ = LabelEncoder()
        self.classes_ = []
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self


def main():
    """ Main """
    print "Main"

if __name__ == "__main__":
    main()
