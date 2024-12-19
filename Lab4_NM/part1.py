import math


def integrand(x):
    return 1 / (x * math.sqrt(9 - x ** 2))


def rectangular_method(a, b, n):
    h = (b - a) / n
    integral_sum = 0

    for i in range(n):
        x = a + (i + 0.5) * h
        integral_sum += integrand(x)

    result = h * integral_sum
    return result


def exact_integral(a, b):
    F_a = (1 / 3) * math.log(a / (3 + math.sqrt(9 - a ** 2)))
    F_b = (1 / 3) * math.log(b / (3 + math.sqrt(9 - b ** 2)))
    return F_b - F_a


a = 1
b = 2
n = 30

approximate_result = rectangular_method(a, b, n)
exact_result = exact_integral(a, b)
epsilon = abs(approximate_result - exact_result) / exact_result

print(f"Method of average rectangles (with {n} partitions): {approximate_result}")
print(f"Exact result (analytical): {exact_result}")
print(f"Eps: {epsilon}")
