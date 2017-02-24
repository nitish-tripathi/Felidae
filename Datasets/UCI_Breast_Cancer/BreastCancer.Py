
""" 
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# Turn class label to 0 or 1
__labeler__ = lambda x: 0 if x == 'M' else 1

def main():
    """ Main """
    df_wdbc = pd.read_csv('wdbc.data', header=None)
    df_wdbc = df_wdbc.drop(df_wdbc.columns[0], axis=1)
    print df_wdbc.shape

    _x_ = df_wdbc.iloc[:, 1:].values
    _y_ = df_wdbc.iloc[:, 0].values
    _y_ = np.array([__labeler__(i) for i in _y_])

    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)
    sc = StandardScaler()
    sc.fit(_x_)
    x_std = sc.transform(_x_)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # PCA analysis of the data
    pca = PCA(n_components=2)
    pca.fit(x_train_std)
    x_train_pca = pca.transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    classifier_ = LogisticRegression(penalty='l1', C=0.1)
    classifier_.fit(x_train_pca, y_train)

    y_pred_test = classifier_.predict(x_test_pca)
    y_pred_train = classifier_.predict(x_train_pca)

    print "Training data PCA Accuracy = %0.02f" % accuracy_score(y_train, y_pred_train)
    print "Test data PCA Accuracy = %0.02f" % accuracy_score(y_test, y_pred_test)

    # LDA analysis of the data
    lda = LDA()
    lda.fit(x_train_std, y_train)
    x_train_lda = lda.transform(x_train_std)
    x_test_lda = lda.transform(x_test_std)

    classifier_ = LogisticRegression(penalty='l1', C=0.1)
    classifier_.fit(x_train_lda, y_train)

    y_pred_test = classifier_.predict(x_test_lda)
    y_pred_train = classifier_.predict(x_train_lda)

    print "Training data LDA Accuracy = %0.02f" % accuracy_score(y_train, y_pred_train)
    print "Test data LDA Accuracy = %0.02f" % accuracy_score(y_test, y_pred_test)

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', n_jobs=2)
    rfc.fit(x_train, y_train)

    y_pred_test = rfc.predict(x_test)
    y_pred_train = rfc.predict(x_train)

    print "Training data Random Forest Accuracy = %0.02f" % accuracy_score(y_train, y_pred_train)
    print "Test data Random Forest Accuracy = %0.02f" % accuracy_score(y_test, y_pred_test)

if __name__ == "__main__":
    main()
