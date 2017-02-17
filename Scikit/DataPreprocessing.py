
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder

def imput_missing_data(data):
    """
    Method that demonstrates imputing missing data
    """
    print data.isnull().sum()
    print data.values
    # Add Missing value using interpolation
    # Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = Imputer()
    imr = imr.fit(data)
    imputed_data = imr.transform(data.values)
    print imputed_data

def handle_categorical_data(data):
    """
    Handle categorical data
    """
    size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1
    }

    #class_mapping = {}
    #for idx, c in enumerate(np.unique(data['class'])):
    #    class_mapping[c] = idx
    class_mapping = {c: idx for idx, c in enumerate(np.unique(data['class']))}

    data['size'] = data['size'].map(size_mapping)
    data['class'] = data['class'].map(class_mapping)

    #print one_hot_encoder(data, 'color')

def one_hot_encoder(data, column_name):
    """
    nominal encoding
    """
    # lambda function
    # input argument:   x - values to be compared
    #                   v - value to compare with
    # returns 1 if x == v else returns 0
    create_boolean_list = lambda x, v: 1 if x == v else 0
    data1 = data.copy()

    for label in np.unique(data[column_name]):
        result = [create_boolean_list(x, label) for x in data[column_name]]
        data1.insert(0, label, result)

    data1.drop(column_name, axis=1, inplace=True)
    return data1

def main():
    """ Main """
    #csv_data = """A,B,C,D
    #              1.0,2.0,,4.0
    #              5.0,6.0,2.3,8.0
    #              0.0,,10.0,11.0"""

    #csv_data = unicode(csv_data)
    #df_ = pd.read_csv(StringIO(csv_data))
    df_ = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df_.columns = ['color', 'size', 'price', 'class']

    handle_categorical_data(df_)
    # imput data
    #imput_missing_data(df_)

if __name__ == "__main__":
    main()
