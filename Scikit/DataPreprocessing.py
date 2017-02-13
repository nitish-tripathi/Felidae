
from io import StringIO
import pandas as pd
from sklearn.preprocessing import Imputer

def main():
    """ Main """
    csv_data = """A,B,C,D
                  1.0,2.0,,4.0
                  5.0,6.0,,8.0
                  0.0,,10.0,11.0"""
    csv_data = unicode(csv_data)
    df_ = pd.read_csv(StringIO(csv_data))

    print df_.isnull().sum()
    print df_.values

    # Add Missing value using interpolation
    # Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = Imputer()
    imr = imr.fit(df_)
    imputed_data = imr.transform(df_.values)
    print imputed_data

if __name__ == "__main__":
    main()
