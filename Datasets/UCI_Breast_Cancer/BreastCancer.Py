
"""
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
"""

import pandas as pd
import numpy as np
import random
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
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve, validation_curve
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

# Turn class label to 0 or 1
__labeler__ = lambda x: 0 if x == 'M' else 1

__colors__ = ['blue', 'red', 'green', 'black', 'yellow', 'cyan', 'magenta']

def plot_3d(X, y):
    """ Plot 3d"""
    data = []

    for i in np.unique(y):
        color = random.choice(__colors__)
        trace = go.Scatter3d(
            x=X[y == i][:, 0],
            y=X[y == i][:, 1],
            z=X[y == i][:, 2],
            mode='markers',
            marker=dict(
                color=color,
                size=3,
                symbol='circle',
            )
        )
        data.append(trace)

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

def main():
    """ Main """

    # Read csv file
    df_wdbc = pd.read_csv('wdbc.data', header=None)
    df_wdbc = df_wdbc.drop(df_wdbc.columns[0], axis=1)
    print df_wdbc.shape

    _x_ = df_wdbc.iloc[:, 1:].values
    _y_ = df_wdbc.iloc[:, 0].values
    _y_ = np.array([__labeler__(i) for i in _y_])

    # Split data into test and training set
    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)

    # Standarize data
    sc = StandardScaler()
    sc.fit(_x_)
    x_std = sc.transform(_x_)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # PCA analysis of the data

    # PCA standard implementation
    pca = PCA(n_components=3)
    pca.fit(x_train_std)
    x_train_pca = pca.transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    plot_3d(x_train_pca, y_train)

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

def train_learning_curve():
    """ Curve """
    df_wdbc = pd.read_csv('wdbc.data', header=None)
    df_wdbc = df_wdbc.drop(df_wdbc.columns[0], axis=1)
    print df_wdbc.shape

    _x_ = df_wdbc.iloc[:, 1:].values
    _y_ = df_wdbc.iloc[:, 0].values
    _y_ = np.array([__labeler__(i) for i in _y_])

    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)

    pipe_lda_lr = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=3)),
        ('lr', LogisticRegression(penalty='l2', random_state=0))
    ])

    # training sizes and its impact on accuracy_score
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lda_lr,
                                                            X=x_train,
                                                            y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=2)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    

    # Impact of changing parameter of LogisticRegression on accuracy_score

    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
        estimator=pipe_lda_lr,
        X=x_train,
        y=y_train,
        param_name='lr__C',
        param_range=param_range,
        cv=10
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', label='Training Accuracy')
    plt.fill_between(param_range,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def grid_search():
    """ Curve """
    df_wdbc = pd.read_csv('wdbc.data', header=None)
    df_wdbc = df_wdbc.drop(df_wdbc.columns[0], axis=1)
    print df_wdbc.shape

    _x_ = df_wdbc.iloc[:, 1:].values
    _y_ = df_wdbc.iloc[:, 0].values
    _y_ = np.array([__labeler__(i) for i in _y_])

    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.2, random_state=0)

    pipe_svc = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=3)),
        ('clf', SVC(kernel='rbf', C=1000, gamma=0.0001, random_state=1))
    ])

    pipe_svc.fit(x_train, y_train)
    print "Test score: %0.02f" % pipe_svc.score(x_test, y_test)
    print "Training score: %0.0f" % pipe_svc.score(x_train, y_train)

    """
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['linear']},
                  {'clf__C': param_range,
                   'clf__gamma': param_range,
                   'clf__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(x_train, y_train)
    print gs.best_score_
    print gs.best_params_
    """

if __name__ == "__main__":
    main()
    #grid_search()
    #train_learning_curve()
