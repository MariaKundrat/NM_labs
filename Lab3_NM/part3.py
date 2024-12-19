def epsilon_algorithm(S, q=2, p=2):
    n = len(S)
    e = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        e[i][0] = S[i]
        if i > 0:
            e[i][1] = 1 / (S[i] - S[i - 1])

    for j in range(2, n):
        for i in range(j, n):
            e[i][j] = e[i - 1][j - 2] + 1 / (e[i][j - 1] - e[i - 1][j - 1])

    epsilon_estimates = [e[i][i] for i in range(n)]
    return epsilon_estimates


S = [1, 1/2, 1/3, 1/4, 1/5]
epsilon_estimates = epsilon_algorithm(S)
print("Epsilon estimates:", epsilon_estimates)
