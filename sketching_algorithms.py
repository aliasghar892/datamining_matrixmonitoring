import numpy as np
import pandas as pd

# this code is based on paper's psudo code (fast frequent directions actully)


def frequent_directions(l, A, B):
    # l is the sketch rows
    # A is the matrix we want to add , it will be a pandas df
    # B is the existing sketch

    # if its the first epoch , i will pass zero matrix to B (size = l*d(number of features))
    for i, a_i in A.iterrows():
        a_i = a_i.values
        # check if B has a row with zeros
        dose_b_has_0 = (B == 0).all(axis=1).any()
        if (not dose_b_has_0):
            # full_matrices=False for optimization
            U, S, Vt = np.linalg.svd(B, full_matrices=False)
            delta = S[-1] ** 2  # in paper it is s[l] meaning last l
            temp = np.sqrt(np.maximum(S ** 2 - delta, 0))
            # matrix multiplaction, np.diag creates a diagonal matrix from an array
            B = pd.DataFrame(np.dot(np.diag(temp), Vt))
        mask = (B == 0).all(axis=1)
        first_zero_index = B.index[mask].tolist()[0]
        B.loc[first_zero_index] = a_i
    return B


def frobenius_norm(df):
    return np.sqrt(np.sum(df**2).sum())


if __name__ == "__main__":
    # test freq  directions
    # Initial matrix B (empty sketch of size l=2, 3 features)
    B = pd.DataFrame(np.zeros((2, 3)))
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    A = pd.DataFrame(data)
    l = 2
    B_updated = frequent_directions(l, A, B)
    print("Updated Sketch Matrix B:")
    print(B_updated)
    print(frobenius_norm(A), frobenius_norm(B_updated))
