
import pandas as pd

def main():
    """
    1. Prepare the data
        a. Check of missing data
        a. Remove Id column
        b. Remove Name column
        c. Convert Sex to one-hot-encoding
        d. Remove Ticket column
        e. Remove Cabin column
    """
    df_titanic = pd.read_csv('train.csv', header=None)
    print df_titanic.describe()

if __name__ == "__main__":
    main()
