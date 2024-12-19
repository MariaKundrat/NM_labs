import math


def f(x):
    return math.cos(x) - 1/(x - 2)


def g(x):
    return 1 / (-2 + x) ** 2 - math.sin(x)


def simple_iteration(x0, eps, max_iterations=100):
    x1 = x0
    iteration = 0
    while iteration < max_iterations:
        x0 = x1
        x1 = g(x0)
        print(f"Iteration {iteration + 1}: x = {x1}")
        if abs(x1 - x0) < eps:
            break
        iteration += 1
    return x1


x0 = -2.5
eps = 1e-6

root = simple_iteration(x0, eps)
print(f"Root of the equation: {root}")
