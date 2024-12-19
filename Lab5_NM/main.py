Umax = 100
f = 50
R1, R2, R3, R4 = 5, 4, 7, 2
L1, L2, L3 = 0.01, 0.02, 0.015
C1, C2, C3 = 300e-6, 150e-6, 200e-6
t_integr = 0.2
h = 0.00001


def input_voltage(t):
    return Umax * np.sin(2 * np.pi * f * t)


def differential_equations(t, y):
    U = input_voltage(t)
    R1, R2, R3, R4 = 5, 4, 7, 2
    L1, L2, L3 = 0.01, 0.02, 0.015
    C1, C2, C3 = 300e-6, 150e-6, 200e-6
    I1, I2, I3, Q1, Q2, Q3 = y


    dI1_dt = (U - R1 * I1 - Q1 / C1) / L1
    dI2_dt = (U - R2 * I2 - Q2 / C2) / L2
    dI3_dt = (U - R3 * I3 - Q3 / C3) / L3


    dQ1_dt = I1
    dQ2_dt = I2
    dQ3_dt = I3


    dy_dt = [dI1_dt, dI2_dt, dI3_dt, dQ1_dt, dQ2_dt, dQ3_dt]
    return dy_dt


def euler_method(dy_dt, y0, t0, t_end, h):
    t_values = np.arange(t0, t_end, h)
    y_values = np.zeros((len(t_values), len(y0)))

    y_values[0] = y0
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + h * dy_dt(t_values[i - 1], y_values[i - 1])

    return t_values, y_values

y0 = [0, 0, 0]

t_values, y_values = euler_method(differential_equations, y0, 0, t_integr, h)

plt.plot(t_values, y_values[:, 1])
plt.xlabel('Час (с)')
plt.ylabel('Напруга U2 (В)')
plt.title('Перехідний процес вихідної напруги U2')
plt.grid(True)
plt.show()
