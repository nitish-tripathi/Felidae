
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

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
    x, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train)
    x_test_std = sc.fit_transform(x_test)

    classifier_ = LogisticRegression(penalty='l1', C=0.1)
    classifier_.fit(x_train_std, y_train)

    print "Training accuracy: %f" % classifier_.score(x_train_std, y_train)
    print "Test accuracy: %f" % classifier_.score(x_test_std, y_test)
    print "Coefficients: %s " % classifier_.coef_

if __name__ == "__main__":
    main()
