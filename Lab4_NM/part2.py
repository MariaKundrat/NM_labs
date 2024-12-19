import numpy as np


def f(x):
    return 1 / np.sqrt(9 - x**2)


def right_rectangle_rule(a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        result += f(a + i * h)
    result *= h
    return result


a = 1
b = 2
n = 30

integral_value = right_rectangle_rule(a, b, n)
print(f"Значення інтегралу: {integral_value}")
