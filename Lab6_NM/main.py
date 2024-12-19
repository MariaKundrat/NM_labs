import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def euler_method_system(function, x_0, t_0, t_n, h):
    num_steps = int((t_n - t_0) / h)
    t_values = np.linspace(t_0, t_n, num_steps + 1)
    x_values = np.zeros((len(t_values), len(x_0)))
    x_values[0] = x_0

    with open('data.txt', 'w') as file:
        for i in range(num_steps):
            x_prev = x_values[i]
            t = t_values[i]

            def equation(x_next):
                return x_prev + h * function(x_next, t + h) - x_next

            x_next_guess = x_prev + h * function(x_prev, t)
            x_next_solution = fsolve(equation, x_next_guess)
            x_values[i + 1] = x_next_solution

            U_C1 = ...
            i_3 = ...
            U_C2 = ...
            L2 = ...
            u1 = ...

        file.write(f"time = {t}, "
                        f"voltage on the capacitor c1 = {U_C1}, "
                        f"current in the inductance = {i_3},"
                        f"voltage on capacitor c2 = {U_C2}\n",
                        f"voltage u2 = {50 * U_C2}, "
                        f"inductance l2 = {L2}, "
                        f"voltage u1 = {u1}\n")

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






















import matplotlib.pyplot as plt

R1 = 1.1
R2 = 2.1
R3 = 50
C1 = 0.15
C2 = 0.17
L_min = 1.5
L_max = 15
i_min = 1
i_max = 2


def runge_kutta_system(x0, y0, x_end, h):
    x_values = []
    U_C1_values = []
    i_3_values = []
    U_C2_values = []
    x = x0
    y1, y2, y3 = y0

    while x < x_end:
        x_values.append(x)
        U_C1_values.append(y1)
        i_3_values.append(y2)
        U_C2_values.append(y3)

        k1 = h * f1(x, y1, y2, y3)
        l1 = h * f2(x, y1, y2, y3)
        m1 = h * f3(x, y1, y2, y3)

        k2 = h * f1(x + h / 2, y1 + k1 / 2, y2 + l1 / 2, y3 + m1 / 2)
        l2 = h * f2(x + h / 2, y1 + k1 / 2, y2 + l1 / 2, y3 + m1 / 2)
        m2 = h * f3(x + h / 2, y1 + k1 / 2, y2 + l1 / 2, y3 + m1 / 2)

        k3 = h * f1(x + h / 2, y1 + 2 * k2 - k1, y2 + 2 * l2 - l1, y3 + 2 * m2 - m1)
        l3 = h * f2(x + h / 2, y1 + 2 * k2 - k1, y2 + 2 * l2 - l1, y3 + 2 * m2 - m1)
        m3 = h * f3(x + h / 2, y1 + 2 * k2 - k1, y2 + 2 * l2 - l1, y3 + 2 * m2 - m1)

        y1 = y1 + (k1 + 4 * k2 + k3) / 6
        y2 = y2 + (l1 + 4 * l2 + l3) / 6
        y3 = y3 + (m1 + 4 * m2 + m3) / 6

        x = x + h

    return x_values, U_C1_values, i_3_values, U_C2_values



def u1(x):
    a = 0.003
    if x % (2 * a) <= a:
        return 10
    else:
        return (-(x % (2 * a)) * 10 / a + 10)


def L2(i_3):
    a0 = (L_min * i_max ** 2 - L_min * i_min ** 2 - L_max *
          i_max ** 2 + L_max * i_min ** 2) / (i_max ** 2 - i_min ** 2)
    a1 = a0 + R1 * i_min
    a2 = a0 + R1 * i_max
    a3 = a0 + R2 * i_max
    if abs(i_3) <= 1:
        return 15
    elif abs(i_3) <= 2:
        return a0 + a1 * (abs(i_3)) + a2 * (abs(i_3) ** 2) + a3 *(abs(i_3) ** 3)
    else:
        return 1.5


def f1(t, U_C1, i_3, U_C2):
    return (u1(t) - U_C1 - U_C2) * 10 ** 3 / (R1 * C1)


def f2(t, U_C1, i_3, U_C2):
    return (U_C2 - i_3 * (R2 + R3)) / L2(i_3)


def f3(t, U_C1, i_3, U_C2):
    return (u1(t) - U_C1 - U_C2 - R1 * i_3) * 10 ** 3 / (R1 * C2)

x0 = 0
y0 = [1, 0, 0]
x_end = 0.03
h = 0.000015
x_values, U_C1_values, i_3_values, U_C2_values = runge_kutta_system(x0, y0, x_end, h)

with open("data.txt", "w") as file:
    for t, U_C1, i_3, U_C2 in zip(x_values, U_C1_values, i_3_values, U_C2_values):
        file.write(f"time = {t}, "
                   f"voltage on the capacitor c1 = {U_C1}, "
                   f"current in the inductance = {i_3},"
                   f"voltage on capacitor c2 = {U_C2},"
                   f"voltage u2 ={50 * U_C2},"
                   f"inductance l2={L2(i_3)},"
                   f"voltage u1={u1(t)}\n")

plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(x_values, U_C1_values)
plt.title("Voltage on capacitor 1 (y1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 2)
plt.plot(x_values, i_3_values)
plt.title("Current in inductor (y2)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 3)
plt.plot(x_values, U_C2_values)
plt.title("Voltage on capacitor 2 (y3)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 4)
plt.plot(x_values, [elem * 50 for elem in U_C2_values])
plt.title("Voltage U2 (50*y3)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 5)
plt.plot(x_values, [u1(elem) for elem in x_values])
plt.title("Voltage U1 (x)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
