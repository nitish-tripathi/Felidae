
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class SBS():
    """
    Class that implement Sequential Backward Feature Selection
    """

    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.indices_ = tuple()
        self.subsets_ = []
        self.scores_ = []
        self.k_score_ = []

    def fit(self, x_data, targets):
        """ Fit to training data """
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, targets, test_size=self.test_size, random_state=self.random_state)

        dim = x_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self._calc_score(x_train, x_test, y_train, y_test, self.indices_)

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            for idx in combinations(self.indices_, r=dim-1):
                score = self._calc_score(x_train, x_test, y_train, y_test, idx)
                scores.append(score)
                subsets.append(idx)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, x_data):
        """ Transform """
        return x_data[:, self.indices_]

    def _calc_score(self, x_train, x_test, y_train, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


def main():
    """ Main """
    # Read data
    df_wine = pd.read_csv("wine.data", header=None)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    print np.unique(df_wine['Class label'])

    # Split data
    _x_, _y_ = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)

    # Standarize the data and transform training and test data
    s_c = StandardScaler()
    x_train_std = s_c.fit_transform(x_train)
    x_test_std = s_c.fit_transform(x_test)

    # Feature reduction using KNN
    classifier_ = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(estimator=classifier_, k_features=1)
    sbs.fit(x_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.show()

    # Accuracy using complete feature set
    classifier_.fit(x_train_std, y_train)
    print "Accuracy with complete trainset: %f" % classifier_.score(x_train_std, y_train)
    print "Accuracy with complete testset: %f" % classifier_.score(x_test_std, y_test)
    
    # Accuracy with selected features
    # sbs.subsets_[8] has only 5 features and has a accuracy of 100%
    #so we will only feautures in sbs.subsets_[8]
    k5 = list(sbs.subsets_[8])
    classifier_.fit(x_train_std[:, k5], y_train)
    print "Accuracy with selected f in trainset: %f" % classifier_.score(x_train_std[:, k5], y_train)
    print "Accuracy with selected f in testset: %f" % classifier_.score(x_test_std[:, k5], y_test)

if __name__ == "__main__":
    main()
