
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as py
import os

def main():
    """ Main """
    # Read dataset
    df_ = pd.read_csv("glass.csv")

    # Get to know the data
    #print df_.shape
    #print df_.head(10)
    #print df_.dtypes

    # Summarize the data
    #print df_.describe()
    #print df_['Type'].value_counts()

    # Visualize the data
    features = df_.columns[:-1].tolist()
    #for feature in features:
        #sns.distplot(df_[feature])
        #py.savefig("visualization/" + feature + ".jpg")
        #plt.show()
    #sns.boxplot(df_[features])
    #plt.show()
    
    sns.pairplot(df_[features], palette='coolwarm')
    plt.show()

if __name__ == "__main__":
    if not os.path.exists("visualization"):
        os.makedirs("visualization")
    main()
