import numpy as np
# import pandas as pd
import numpy as np
import math

# this code is based on paper's psudo code (fast frequent directions actully)


def frequent_directions(l, A, B):
    # l is the sketch rows
    # A is the matrix we want to add (numpy array)
    # B is the existing sketch (numpy array)

    # if its the first epoch , i will pass zero matrix to B (size = l*d(number of features))
    for i in range(A.shape[0]):
        a_i = A[i, :]
        # check if B has a row with zeros
        dose_b_has_0 = np.any(np.all(B == 0, axis=1))
        # if not dose_b_has_0:
        freq_try = 0
        while not dose_b_has_0:
            if freq_try > 0:
                # print(f"try {freq_try} for frequent directions")
                pass
            # full_matrices=False for optimization
            U, S, Vt = np.linalg.svd(B, full_matrices=False)
            delta = S[-1] ** 2  # in paper it is s[l] meaning last l
            temp = np.sqrt(np.maximum(S ** 2 - delta, 0))
            # matrix multiplication, np.diag creates a diagonal matrix from an array
            B = np.dot(np.diag(temp), Vt)
            dose_b_has_0 = np.any(np.all(B == 0, axis=1))
            freq_try += 1

        # Find first row with all zeros
        zero_rows = np.where(np.all(B == 0, axis=1))[0]
        if len(zero_rows) > 0:
            first_zero_index = zero_rows[0]
            B[first_zero_index] = a_i
        else:
            raise Exception("no zero row !")

    return B


def mse(matrix1, matrix2):
    return np.mean((matrix1 - matrix2) ** 2)


# calculate k with Johnson-Lindenstrauss lemma with respect to error e
def calc_k_for_gaussian(rows, e=0.5):
    return int(np.ceil((4 * math.log(rows)) / ((e**2 / 2) - (e**3 / 3))))


def gaussian_random_projection(X, k):
    # gaussian random projection
    rows, num_features = X.shape

    if (k > num_features):  # gaussian projection is useless in this case
        print("gaussian random projection is useless in this case , k provided was ", k)
        return X

    # create random gaussian matrix with mean=0, variance=1/k
    R = np.random.normal(loc=0.0, scale=1/np.sqrt(k), size=(num_features, k))
    X_projected = np.dot(X, R)
    # if X is m*n and R is n*k then new_x (m*k) * R transpose (k*n) will have shape m*n , its just approximation
    x_reconstructed = np.dot(X_projected, R.T)
    reconstruction_error = mse(X, x_reconstructed)
    return X_projected, reconstruction_error


def frobenius_norm(df):
    return np.sqrt(np.sum(df**2, axis=0).sum())


def total_variance(X, center=True):
    # i use this to show the total variance captured by each sketch
    # we center because we need vairiance around the mean not zero
    if center:
        X = X - np.mean(X, axis=0)  # center the data

    cov_matrix = np.cov(X, rowvar=False)  # shape: (num_features, num_features)
    # # efficient for symmetric matrices
    # eigenvalues = np.linalg.eigvalsh(cov_matrix)
    # in any symmetric matrix trace is equal to sum of eigen values
    total_variance = np.trace(cov_matrix)
    return total_variance


# based on algorithm 1 of the paper
# def compute_anomaly_scores(data, sketch, k):
#     U_sketch, S_sketch, Vt_sketch = np.linalg.svd(sketch, full_matrices=False)
#     V_k = Vt_sketch[:k].T
#     S_k = S_sketch[:k]
#     A_Vk = np.dot(data, V_k)
#     leverage_scores = np.sum((A_Vk / S_k)**2, axis=1)
#     V_remaining = Vt_sketch[k:].T
#     A_V_remaining = np.dot(data, V_remaining)
#     projection_distances = np.sum(A_V_remaining**2, axis=1)
#     return leverage_scores, projection_distances


# THIS ONE IS SAME AS ABOVE BUT ACTS BETTER
def compute_anomaly_scores(data, sketch, k):
    # data is the new data
    # sketch is the sketch before updating it with data (data will be leaked into sketch if done after updating !)
    # k is the top k right singular and singular values
    U_sketch, S_sketch, Vt_sketch = np.linalg.svd(sketch, full_matrices=False)
    V_k = Vt_sketch[:k].T
    S_k = S_sketch[:k]
    A_Vk = np.dot(data, V_k)
    leverage_scores = np.sum((A_Vk / S_k)**2, axis=1)

    row_norms_squared = np.sum(data**2, axis=1)  # norm 2 squared of data
    proj_norms_squared = np.sum(A_Vk**2, axis=1)
    projection_distances = row_norms_squared - proj_norms_squared

    return leverage_scores, projection_distances


class IPCA:
    def __init__(self, n_components=None):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.cov = None
        self.components = None
        self.eigvals = None
        self.n_components = n_components

    def partial_fit(self, x):
        # update the mean and covariance incrementally with a new sample x
        if self.mean is None:  # fisrst
            self.n = 1
            self.mean = x.copy()
            self.M2 = np.zeros((x.size, x.size))
            self.cov = np.zeros_like(self.M2)
            return self

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        # welford's online covariance update
        self.M2 += np.outer(delta, x - self.mean)
        if self.n > 1:
            self.cov = self.M2 / (self.n - 1)
        else:
            self.cov = np.zeros_like(self.M2)
        return self

    def compute_pca(self):
        # make it symmetric if it is not due to error
        self.cov = (self.cov + self.cov.T) / 2
        # self.cov += np.eye(self.cov.shape[0]) * 1e-12
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        idx = np.argsort(eigvals)[::-1]  # descending order
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        if self.n_components is not None:
            eigvecs = eigvecs[:, :self.n_components]
            eigvals = eigvals[:self.n_components]

        self.eigvals = eigvals
        self.components = eigvecs
        return eigvals, eigvecs

    def transform(self, x):
        if self.components is None:
            self.compute_pca()
        x_centered = x - self.mean
        z = np.dot(x_centered, self.components)
        return z

    def inverse_transform(self, z):
        x_rec = self.mean + np.dot(z, self.components.T)
        return x_rec

    def step(self, x):
        self.partial_fit(x)
        if self.n % 50 == 1:  # update every 50 samples, soooo faster this way
            self.compute_pca()
        # eigvals, eigvecs = self.compute_pca()

        total_var = np.sum(self.eigvals)
        projected = self.transform(x)
        reconstructed = self.inverse_transform(projected)
        reconstruction_error = mse(x, reconstructed)
        # reson used cumsum: first component has ... percent , second and first has ...
        var_ratio = np.cumsum(self.eigvals) / (total_var)
        return projected,  reconstruction_error, var_ratio


if __name__ == "__main__":
    B = np.zeros((2, 3))
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    l = 2
    B_updated = frequent_directions(l, A, B)
    print("Updated Sketch Matrix B:")
    print(B_updated)
    print(frobenius_norm(A), frobenius_norm(B_updated))

    rand_data = np.random.randn(500, 1000)
    project, reconstruction_error = gaussian_random_projection(
        rand_data, k=calc_k_for_gaussian(rand_data.shape[0]))
    print(calc_k_for_gaussian(rand_data.shape[0]))
    print("test projection:")
    print("reconstruction error: ", reconstruction_error)
    # print(project)
    print(frobenius_norm(project), frobenius_norm(rand_data))

#     Updated Sketch Matrix B:
#           0         1         2
# 0 -4.062293 -5.366646 -6.670999
# 1  7.000000  8.000000  9.000000
# 16.881943016134134 16.84652323337097
