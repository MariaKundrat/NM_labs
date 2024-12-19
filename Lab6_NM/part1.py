import matplotlib.pyplot as plt

R1 = 170
R2 = 23
R3 = 70
C1 = 0.3
C2 = 0.5
L1 = 18.8
L_min = 0.03
L_max = 0.3
i_min = 1
i_max = 2


def runge_kutta_system(x0, y0, x_end, h):
    x_values = []
    U_C1_values = []
    i_3_values = []
    U_C2_values = []

    x = x0
    U_C1, i_3, U_C2 = y0

    while x < x_end:
        x_values.append(x)
        U_C1_values.append(U_C1)
        i_3_values.append(i_3)
        U_C2_values.append(U_C2)
        k1 = h * f1(x, U_C1, i_3, U_C2)
        l1 = h * f2(x, U_C1, i_3, U_C2)
        m1 = h * f3(x, U_C1, i_3, U_C2)
        k2 = h * f1(x + h / 2, U_C1 + k1 / 2, i_3 + l1 / 2, U_C2 + m1 / 2)
        l2 = h * f2(x + h / 2, U_C1 + k1 / 2, i_3 + l1 / 2, U_C2 + m1 / 2)
        m2 = h * f3(x + h / 2, U_C1 + k1 / 2, i_3 + l1 / 2, U_C2 + m1 / 2)
        k3 = h * f1(x + h / 2, U_C1 + 2 * k2 - k1, i_3 + 2 * l2 - l1, U_C2 + 2 * m2 - m1)
        l3 = h * f2(x + h / 2, U_C1 + 2 * k2 - k1, i_3 + 2 * l2 - l1, U_C2 + 2 * m2 - m1)
        m3 = h * f3(x + h / 2, U_C1 + 2 * k2 - k1, i_3 + 2 * l2 - l1, U_C2 + 2 * m2 - m1)
        U_C1 = U_C1 + (k1 + 4 * k2 + k3) / 6
        i_3 = i_3 + (l1 + 4 * l2 + l3) / 6
        U_C2 = U_C2 + (m1 + 4 * m2 + m3) / 6

        x = x + h

    return x_values, U_C1_values, i_3_values, U_C2_values


def u1(x):
    a = 0.003
    if x % (2 * a) <= a:
        return 10
    else:
        return (-(x % (2 * a)) * 10 / a + 10)


def L2(i_3):
    a0 = (L_min * i_max ** 2 - L_min * i_min ** 2 - L_max * i_max ** 2 + L_max * i_min ** 2) / (i_max ** 2 - i_min ** 2)
    a1 = a0 + R1 * i_min
    a2 = a0 + R1 * i_max
    a3 = a0 + R2 * i_max

    if abs(i_3) <= 1:
        return 15
    elif abs(i_3) <= 2:
        return a0 + a1 * (abs(i_3)) + a2 * (abs(i_3) ** 2) + a3 * (abs(i_3) ** 3)
    else:
        return 1.5


def f1(t, U_C1, i_3, U_C2):

    return (u1(t) - U_C1 - U_C2 + U_C2 * R3 - U_C2 * R2) * 10 ** 3 / ((R1 + R3 - R2) * (C1))


def f2(t, U_C1, i_3, U_C2):

    return (u1(t) - U_C1 - U_C2 + U_C2 * R3 - U_C2 * R2 - U_C2 * (R1 + R3 - R2)) * 10 ** 3 / ((R1 + R3 - R2) * (C2))


def f3(t, U_C1, i_3, U_C2):

    return U_C1 + ((u1(t) - U_C2 - U_C1 + U_C1 * R3 - U_C1 * R2 - U_C1 * (R1 + R3 - R2) * (R3 - R2)) * 10 ** 3 / (
            (R1 + R3 - R2) * (L1)))


x0 = 0
y0 = [1, 0, 0]
x_end = 0.03
h = 0.000015
x_values, U_C1_values, i_3_values, U_C2_values = runge_kutta_system(x0, y0, x_end, h)

with open("data.txt", "w", encoding="utf-8") as file:
    for t, U_C1, i_3, U_C2 in zip(x_values, U_C1_values, i_3_values, U_C2_values):
        file.write(f"time = {t}, "
                   f"voltage on capacitor c1 = {U_C1},"
                   f"current in the inductance = {i_3},"
                   f"voltage on capacitor c2 = {U_C2},"
                   f"voltage u2 ={R3 * i_3},"
                   f"inductance l2={L2(i_3)},"
                   f"voltage u1={u1(t)}\n")

plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(x_values, U_C1_values)
plt.title("Voltage on capacitor 1 (U_C1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 2)
plt.plot(x_values, i_3_values)
plt.title("Current in the inductor (i_3)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 3)
plt.plot(x_values, U_C2_values)
plt.title("Voltage on capacitor 2 (U_C2)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 4)
plt.plot(x_values, [elem * 50 for elem in U_C2_values])
plt.title("Voltage U2 (R3*i_3)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.subplot(3, 2, 5)
plt.plot(x_values, [u1(elem) for elem in x_values])
plt.title("Voltage U1 (x)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
