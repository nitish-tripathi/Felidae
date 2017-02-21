
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd

def main():
    """ Main """
    """
    names_ = ['animal_name',
              'hair',
              'feathers',
              'eggs',
              'milk',
              'airborne',
              'aquatic',
              'predator',
              'toothed',
              'backbone',
              'breathes',
              'venomous',
              'fins',
              'legs',
              'tail',
              'domestic',
              'catsize',
              'class_type']
    """
    data_ = pd.read_csv('zoo.csv')
    _x_ = data_.iloc[:, 1:-1].values
    _y_ = data_.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(_x_, _y_, test_size=0.25, random_state=0)

    _classifier = RandomForestClassifier(
        criterion='entropy',
        n_estimators=10,
        random_state=0,
        n_jobs=2)
    _classifier.fit(x_train, y_train)

    y_pred = _classifier.predict(x_test)

    print "Accuracy: %f" % accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    main()
