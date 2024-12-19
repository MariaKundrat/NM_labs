import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def euler_method_system(function, x_0, t_0, t_n, h):
    num_steps = int((t_n - t_0) / h)
    t_values = np.linspace(t_0, t_n, num_steps + 1)
    x_values = np.zeros((len(t_values), len(x_0)))
    x_values[0] = x_0

    for i in range(num_steps):
        x_prev = x_values[i]
        t = t_values[i]

        def equation(x_next):
            return x_prev + h * function(x_next, t + h) - x_next

        x_next_guess = x_prev + h * function(x_prev, t)
        x_next_solution = fsolve(equation, x_next_guess)
        x_values[i + 1] = x_next_solution

    return t_values, x_values


def system_equations(x, t):
    u = 100 * math.sin(2 * math.pi * 50 * t)
    dxdt = (u - x[0] - x[1] + x[1] * 7 - x[1] * 4) / ((5 + 7 - 4) * 300e-6)
    dydt = (u - x[0] - x[1] + x[1] * 7 - x[1] * 4 - x[1] * (5 + 7 - 4)) / ((5 + 7 - 4) * 300e-6)
    dzdt = x[1] + ((u - x[0] - x[1] + x[1] * 7 - x[1] * 4 - x[1] * (5 + 7 - 4) * (7 - 4)) / ((5 + 7 - 4) * 0.01))
    return np.array([dxdt, dydt, dzdt])


def plot_graphs(t_values, u1_values, u2_values):
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, u1_values, label='u1')
    plt.plot(t_values, u2_values, label='u2')
    plt.xlabel('Time (t)')
    plt.ylabel('Value')
    plt.legend()
    plt.title('System of differential equations')
    plt.grid(True)
    plt.show()


t_values, x_values = euler_method_system(function=system_equations, x_0=np.array([0, 0, 0]), t_0=0, t_n=0.2, h=1e-6)

u2_values = [4 * elem for elem in x_values[:, 2]]
u1_values = [100 * math.sin(2 * math.pi * 50 * t) for t in t_values]

plot_graphs(t_values, u1_values, u2_values)

for t, x in zip(t_values, x_values):
    print(f"t = {t:.2f}, u2 = {4 * x[2]:.4f}")
