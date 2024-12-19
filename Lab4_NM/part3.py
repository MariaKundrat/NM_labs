import numpy as np


def f(x):
    return 1 / (x * np.sqrt(9 - x ** 2))

a = 1
b = 2

n = 100

h = (b - a) / n

x = np.linspace(a, b, n, endpoint=False)

y = f(x)

print(x)
print(y)

S = h * y

I = np.sum(S)

def F(x):
    return (1 / 3) * np.log(x / (3 + np.sqrt(9 - x ** 2)))

exact = F(b) - F(a)

e = abs(I - exact) / exact

print("Наближене значення інтегралу:")
print(I)
print("Точне значення інтегралу:")
print(exact)
print("Похибка:")
print(e)
