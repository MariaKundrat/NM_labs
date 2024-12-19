import numpy as np


def system_equations(x):
    f1 = x[0] ** 2 * x[1] ** 2 - (x[0] ** 2 - x[1] ** 2) ** 2 - 0.5 - x[0]
    f2 = x[0] * x[1] * ((x[1] ** 2 - x[0] ** 2) - 0.5 - x[1])
    return np.array([f1, f2])


def jacobian_matrix(x):
    h = 1e-5
    df1_dx1 = (system_equations([x[0] + h, x[1]])[0] - system_equations(x)[0]) / h
    df1_dx2 = (system_equations([x[0], x[1] + h])[0] - system_equations(x)[0]) / h
    df2_dx1 = (system_equations([x[0] + h, x[1]])[1] - system_equations(x)[1]) / h
    df2_dx2 = (system_equations([x[0], x[1] + h])[1] - system_equations(x)[1]) / h

    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])


def solve_system(initial_guess, tolerance, max_iterations=1000):
    x = np.array(initial_guess)
    iteration = 0

    while iteration < max_iterations:
        f_values = system_equations(x)
        jacobian = jacobian_matrix(x)

        try:
            inverted_jacobian = np.linalg.inv(jacobian)
        except np.linalg.LinAlgError:
            break

        h = np.dot(inverted_jacobian, -f_values)
        x = x + h

        if np.linalg.norm(h) < tolerance * np.linalg.norm(x):
            yield x, iteration
            break

        iteration += 1
        yield x, iteration


initial_guess = [1, 2]
tolerance = 1e-5

last_result = None
last_iterations = None

solver = solve_system(initial_guess, tolerance)
for result, iterations in solver:
    print(f"Result: x = {result}, Counter: {iterations}")
    last_result = result
    last_iterations = iterations

print(f"\nGeneral result: x = {last_result}, Counter: {last_iterations}")
