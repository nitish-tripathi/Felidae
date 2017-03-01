
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.externals import six
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import numpy as np

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

    def predict(self, X):
        """ Predict """
        if self.vote == 'classlabel':
            # Collect predictions from every Classifier
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                                                                           weights=self.weights)),
                                           axis=1,
                                           arr=predictions)
        elif self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),
                                 axis=1)

        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict probability """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_probas = np.average(probas, axis=0, weights=self.weights)
        return avg_probas

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier,
                         self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in\
                    six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(
                        step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

        return out

def main():
    """ Main """
    print "Main"

if __name__ == "__main__":
    main()
