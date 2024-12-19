import numpy as np


def gauss_inverse(matrix):
    n = len(matrix)
    inverse = np.identity(n)

    for i in range(n):
        max_index = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[max_index][i]):
                max_index = j

        matrix[[i, max_index]] = matrix[[max_index, i]]
        inverse[[i, max_index]] = inverse[[max_index, i]]

        main_elem = matrix[i][i]
        matrix[i] /= main_elem
        inverse[i] /= main_elem

        for j in range(n):
            if i != j:
                factor = matrix[j][i]
                matrix[j] -= factor * matrix[i]
                inverse[j] -= factor * inverse[i]

    return inverse


def system_matrix(s, B, k, p):
    matrix = np.array([[8.3, 2.62 + s, 4.1, 1.9], [3.92, 8.45, 7.78 - s, 2.46], [3.77, 7.21 + s, 8.04, 2.28],
                       [2.21, 3.65 - s, 1.69, 6.69]])
    return matrix


matrix = system_matrix(0.02, 0.02 * 25, 10, 25)
print(matrix)
matrix_a = np.copy(matrix)
inverse_matrix = gauss_inverse(matrix)
print("Inverse matrix:\n", inverse_matrix)
print(matrix_a)
a = np.matmul(matrix_a, inverse_matrix)
print(a)
