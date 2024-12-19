import numpy as np


def secant_method(f, x0, x1, eps, max_iter):
    iter_count = 0
    while abs(x1 - x0) > eps and iter_count < max_iter:
        if f(x1) - f(x0) == 0:
            return None
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
        iter_count += 1
    return x1 if iter_count < max_iter else None


def gauss_with_pivoting(A, b):
    n = len(A)
    M = np.hstack((A, b.reshape(-1, 1)))

    for k in range(n):
        maxindex = abs(M[k:, k]).argmax() + k
        if M[maxindex, k] == 0:
            return None
        M[[k, maxindex]] = M[[maxindex, k]]
        for i in range(k + 1, n):
            if M[k, k] == 0:
                return None
            f = M[i, k] / M[k, k]
            M[i, k:] -= f * M[k, k:]

    # Зворотній хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = M[i, -1]
        for j in range(i + 1, n):
            x[i] -= M[i, j] * x[j]
        if M[i, i] == 0:
            raise ValueError("Ділення на нуль - матриця сингулярна.")
        x[i] /= M[i, i]

    return x


def f1(x1, x2):
    return x1 ** 2 * x2 ** 2 - (x2 - x1) ** 2 / 4 - 0.5 * x1


def f2(x1, x2):
    return x1 * x2 ** 2 - x2 - 0.5 * x1


x1_0 = 1.0
x2_0 = 1.0
eps = 1e-5
max_iter = 100


root_x1 = secant_method(lambda x: f1(x, x2_0), x1_0, x1_0 + eps, eps, max_iter)
root_x2 = secant_method(lambda x: f2(root_x1, x), x2_0, x2_0 + eps, eps, max_iter)

if root_x1 is not None and root_x2 is not None:
    print(f"Корені рівнянь методом січних: x1 = {root_x1}, x2 = {root_x2}")
else:
    print("Не вдалося знайти корені методом січних.")


A = np.array([[3, 2, -4], [2, 3, 3], [5, -3, 1]], dtype=float)
b = np.array([1, 2, 3], dtype=float)


solution = gauss_with_pivoting(A, b)
if solution is not None:
    print(f"Розв'язок системи рівнянь методом Гауса: {solution}")
else:
    print("Не вдалося знайти розв'язок методом Гауса.")

