import numpy as np


def F(x):
    return np.array([x[0]**2 - x[1] + 1, x[0] + x[1]**2 - 3])


def jacobian_approx(f, x, h=1e-5):
    n = len(x)
    J = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        x[i] += h
        f1 = f(x)
        x[i] -= h
        J[:, i] = (f1 - f0) / h
    return J


def newton_method(F, x0, tol=1e-5, max_iter=100):
    x = x0
    for _ in range(max_iter):
        J = jacobian_approx(F, x)
        Fx = F(x)
        dx = np.linalg.solve(J, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            return x
    return x


x0 = np.array([1.0, 1.0])


root = newton_method(F, x0)
print(f"Корінь системи рівнянь: {root}")
