
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
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

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

    # Pipe implementation
    pipe_pca = Pipeline([('sc', StandardScaler()),
                         ('pca', PCA(n_components=2)),
                         ('lr', LogisticRegression())])

    scores = []
    kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    for k, (train, test) in enumerate(kfold):
        pipe_pca.fit(x_train[train], y_train[train])
        score = pipe_pca.score(x_train[test], y_train[test])
        scores.append(score)
        print "Fold: %d, Acc: %0.02f" % (k, score)

    # PCA standard implementation
    pca = PCA(n_components=3)
    pca.fit(x_train_std)
    x_train_pca = pca.transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    trace1 = go.Scatter3d(
        x=x_train_pca[y_train == 0][:, 0],
        y=x_train_pca[y_train == 0][:, 1],
        z=x_train_pca[y_train == 0][:, 2],
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=3,
            symbol='circle',
        )
    )

    trace2 = go.Scatter3d(
        x=x_train_pca[y_train == 1][:, 0],
        y=x_train_pca[y_train == 1][:, 1],
        z=x_train_pca[y_train == 1][:, 2],
        mode='markers',
        marker=dict(
            color='rgb(254, 0, 0)',
            size=3,
            symbol='circle'
        )
    )

    data = [trace1, trace2]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

    #plt.scatter(x_train_pca[y_train == 0][:, 0], x_train_pca[y_train == 0][:, 1], color='red')
    #plt.scatter(x_train_pca[y_train == 1][:, 0], x_train_pca[y_train == 1][:, 1], color='blue')
    #plt.xlabel("PCA1")
    #plt.ylabel("PCA2")
    #plt.show()

    lr_ = LogisticRegression()
    lr_.fit(x_train_pca, y_train)

    y_pred_test = lr_.predict(x_test_pca)
    y_pred_train = lr_.predict(x_train_pca)

    print "Training data PCA Accuracy = %0.02f" % accuracy_score(y_train,
                                                                 y_pred_train)
    print "Test data PCA Accuracy = %0.02f" % accuracy_score(y_test,
                                                             y_pred_test)
    print "----"

    svm = SVC(kernel='rbf', gamma=0.1, C=2)
    svm.fit(x_train_std, y_train)

    y_pred_test = svm.predict(x_test_std)
    y_pred_train = svm.predict(x_train_std)

    print "Training data SVM Accuracy = %0.02f" % accuracy_score(y_train,
                                                                 y_pred_train)
    print "Test data SVM Accuracy = %0.02f" % accuracy_score(y_test,
                                                             y_pred_test)
    print "----"

    # LDA analysis of the data
    lda = LDA(solver='eigen')
    lda.fit(x_train_std, y_train)
    x_train_lda = lda.transform(x_train_std)
    x_test_lda = lda.transform(x_test_std)

    classifier_ = LogisticRegression()
    classifier_.fit(x_train_lda, y_train)

    y_pred_test = classifier_.predict(x_test_lda)
    y_pred_train = classifier_.predict(x_train_lda)

    print "Training data LDA Accuracy = %0.02f" % accuracy_score(y_train, y_pred_train)
    print "Test data LDA Accuracy = %0.02f" % accuracy_score(y_test, y_pred_test)
    print "----"

    zeros = np.zeros(x_train_lda.shape)
    plt.scatter(x_train_lda[y_train == 0][:, 0], zeros[y_train == 0][:, 0], color='red')
    plt.scatter(x_train_lda[y_train == 1][:, 0], zeros[y_train == 1][:, 0], color='blue')
    plt.xlabel('LDA1')
    plt.show()

    # Random Forest Classifier
    """
    rfc = RandomForestClassifier(n_estimators=100,
                                 min_samples_split=25,
                                 max_depth=7,
                                 max_features=2)
    rfc.fit(x_train, y_train)

    y_pred_test = rfc.predict(x_test)
    y_pred_train = rfc.predict(x_train)

    print "Training data Random Forest Accuracy = %0.02f" % accuracy_score(y_train, y_pred_train)
    print "Test data Random Forest Accuracy = %0.02f" % accuracy_score(y_test, y_pred_test)
    """

if __name__ == "__main__":
    main()
