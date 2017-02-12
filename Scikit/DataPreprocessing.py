
from io import StringIO
import pandas as pd

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

if __name__ == "__main__":
    main()
