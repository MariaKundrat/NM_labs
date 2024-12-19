import math


def f(x):
    return math.cos(x) - 1/(x - 2)


def df(x):
    return -math.sin(x) - 1/((x - 2)**2)


def chord_method(f, x0, x1, tol):
    while abs(x1 - x0) > tol:
        x0, x1 = x1, x1 - (x1 - x0) * f(x1) / (f(x1) - f(x0))
    return x1


def newton_method(f, df, x0, tol):
    while True:
        x1 = x0 - f(x0)/df(x0)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1


def combined_method(f, df, a, b, tol):
    if f(a) * df(a) > 0:
        return chord_method(f, b, a, tol), newton_method(f, df, b, tol)
    else:
        return chord_method(f, a, b, tol), newton_method(f, df, a, tol)


a, b = -2.5, -0.5
tol = 1e-5

chord_root, newton_root = combined_method(f, df, a, b, tol)
print(f"Корінь рівняння за методом хорд: {chord_root}")
print(f"Корінь рівняння за методом дотичних (Ньютона): {newton_root}")
