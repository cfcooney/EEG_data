# Convert labels to one-hot encoded values
def one_hot(y):
    y = y.rehape(len(y))
    n_values = np.max(y)+1
    return np.eye(n_values)[np,array(y, dtype=np.int32)]

"""
Converts exponential numbers to floating point values.
e.g. 6.65714259e-04 to 0.00066571.
"""
def convert_to_float(value):
    converted_numbers = []
    for s in value:
        s = str(s)
        value2 = float("{:.8f}".format(float(s)))
        converted_numbers.append(value2)
    return converted_numbers

#Reduce a large dataset to allow testing on small subset.
def reduce_data(fin):
    count = 0
    reduced_set = []
    for s in fin:
        if count < len(fin)/100: #Number '100' can be chnaged depending on requirements
            reduced_set.append(s)
            count += 1
    return reduced_set