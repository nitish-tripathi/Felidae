
"""
My implementation of Decision Tree
"""
from math import log

def divide_set(data_set, column, value):
    """
    Divides a set on a specific column. Can handle numeric or nominal values
    """
    # Make a function that tells us if a row is in the first group (true)
    # or the second group (false)
    split_function = None

    # check if the value is a number i.e int or float
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Divide the rows into two sets and return them
    set1 = [row for row in data_set if split_function(row)]
    set2 = [row for row in data_set if not split_function(row)]

    return (set1, set2)

def unique_counts(data_set):
    """ Gets the unique counts of the number each target class """
    results = {}
    for feature in data_set:
        target = feature[-1]
        if target not in results:
            results[target] = 0
        results[target] += 1
    return results

def entropy(data_set):
    """
    Calculate entropy of the data_set
    Entropy is the sum of p(x)log(p(x)) across all the different possible results
    """
    log2 = lambda x: log(x)/log(2)
    results = unique_counts(data_set)
    ent = 0.0
    for count in results.keys():
        ratio = float(results[count])/len(data_set)
        ent = ent-ratio*log2(ratio)

    return ent

def main():
    """ Main """
    my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
               ['google', 'France', 'yes', 23, 'Premium'],
               ['digg', 'USA', 'yes', 24, 'Basic'],
               ['kiwitobes', 'France', 'yes', 23, 'Basic'],
               ['google', 'UK', 'no', 21, 'Premium'],
               ['(direct)', 'New Zealand', 'no', 12, 'None'],
               ['(direct)', 'UK', 'no', 21, 'Basic'],
               ['google', 'USA', 'no', 24, 'Premium'],
               ['slashdot', 'France', 'yes', 19, 'None'],
               ['digg', 'USA', 'no', 18, 'None'],
               ['google', 'UK', 'no', 18, 'None'],
               ['kiwitobes', 'UK', 'no', 19, 'None'],
               ['digg', 'New Zealand', 'yes', 12, 'Basic'],
               ['slashdot', 'UK', 'no', 21, 'None'],
               ['google', 'UK', 'yes', 18, 'Basic'],
               ['kiwitobes', 'France', 'yes', 19, 'Basic']]

    x_1, x_2 = divide_set(my_data, 3, 20)
    print entropy(my_data)
    print entropy(x_1)
    print entropy(x_2)

if __name__ == "__main__":
    main()
