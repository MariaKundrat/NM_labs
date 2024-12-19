import math


def f(x):
    return math.cos(x) - 1 / (x - 2)


def steffensen(f, x0, tol):
    def g(x):
        return x - f(x) ** 2 / (f(x + f(x)) - f(x))

    x1 = g(x0)
    while abs(x1 - x0) > tol:
        x0, x1 = x1, g(x1)
    return x1


x0 = -2.5
tol = 1e-5

root = steffensen(f, x0, tol)
print(f"Корінь рівняння: {root}")
