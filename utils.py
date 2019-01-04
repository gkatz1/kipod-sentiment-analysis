import numpy as np


# How to generate batch?
# Do we want to naively iterate through all examples
# Or maybe sample uniformly from each class
def batch_generator(data, batch_size):
    """
    Generates the next batch
    """
    X, y = data

    if X.shape[0] != y.shape[0]:
        raise Exception("non matching dimensions for X ({}) and y ({})".format(
            X.shape[0], y.shape[0]))

    size = X.shape[0]
    
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            if i < size:
                to_yield = X[i:size], y[i:size]
                num_left_to_yield = batch_size - (size - i)
                
                i = 0
                yield (np.concatenate((to_yield[0], X[i:i + num_left_to_yield]), axis=0),
                    np.concatenate((to_yield[1], y[i:i + num_left_to_yield]), axis=0))
                i += num_left_to_yield

            else:
                i = 0


def load_word_vectors(path):
    word_vectors = np.load(path)
    return word_vectors


def find_first_occ(arr, target_val):
    """
    Returns first occurrence of target_val in arr
    
    :param arr: array to search target_val in
    :param target_val: value to search its first occurrence in arr
    :return: index of 1st occurrence of target_val in arr (if no such occurrence return -1)
    """
    for idx, val in enumerate(arr):
        if val == target_val:
            return idx

    return -1


def get_lengths(X, padd_value):
    """
    Returns the lengths of the unpadded vector for each vector in X
    
    :param X: array of arrays to calculate their lengths
    :param padd_value: padding value
    :return: numpy array of lengths of unpadded parts in arrays of X
    """
    return np.argmax(X == 0, axis=1)

    lengths = []
    for x in X:
        # length = find_first_occ(x, padd_value)
        length = np.argmax(x == 0)
        if length == 0 and x[0] != 0:
        # if length == -1:
            length = max(x.shape)
        lengths.append(length)

    return np.array(lengths)


