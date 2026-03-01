import numpy as np
from scipy import stats
import pandas as pd

# data = pd.read_csv("C:/Users/ucg8nb/Downloads/3062-2026-hw-2-problem-2.csv")

# n = np.array(data['n'].tolist())
# X = np.array(data['X'].tolist())
# Y = np.array(data['Y'].tolist())

# ci_t_x = stats.t.interval(0.95, df = len(X) - 1, loc = X.mean(),  scale = stats.sem(X))

# ci_t_y = stats.t.interval(0.95, df = len(Y) - 1, loc = Y.mean(), scale = stats.sem(Y))

# print(ci_t_x)
# print(ci_t_y)

# ci_z_x = stats.norm.interval(0.95, loc = X.mean(), scale = X.std() / np.sqrt(len(X)))
# ci_z_y = stats.norm.interval(0.95, loc = Y.mean(), scale = Y.std() / np.sqrt(len(Y)))

# print(ci_z_x)
# print(ci_z_y)

# Z = [1.2, 1.5, 1.68, 1.89, 0.95, 1.49, 1.58, 1.55, 0.5, 1.09]
# Z = np.array(Z)


# n = len(Z)
# xBar = Z.mean()
# std = Z.std()
# s2 = np.var(Z, ddof = 1)

# se = np.sqrt(s2 / n)

# z_crit = stats.norm.ppf(1 - 0.05 * 0.5, loc = 0, scale = 1)

# t_crit = stats.t.ppf(1 - 0.05 * 0.5, df = n  -1 )

# print(z_crit * se)

# print(xBar - z_crit * se, xBar + z_crit * se)
# print(xBar - t_crit * se, xBar + t_crit * se)

import numpy as np
from scipy.stats import norm
import math

def MCE(g, dist, beta):
    alpha = 0.01
    z = norm.ppf(1 - alpha/2)

    n = 0
    mean = 0
    M2 = 0

    while True:
        x = dist.rvs()
        y = g(x)

        n += 1
        delta = y - mean
        mean += delta / n
        delta2 = y - mean
        M2 += delta * delta2

        if n > 1:
            s2 = M2 / (n - 1)

            halfwidth = z * np.sqrt(s2 / n)

            if halfwidth <= beta:
                return mean, n, halfwidth
            
def MCETest(f, dist, beta):
    alpha = 0.01
    z = norm.ppf(1 - alpha/2)

    n = 0

    y_vals = []

    while True:
        n += 1
        if n == 1:
            for _ in range(10):

                x = dist.rvs()
                y = f(x)
                n += 1
                y_vals.append(y)
        else:
            x = dist.rvs()
            y = f(x)
            y_vals.append(y)
        s = np.std(y_vals, ddof = 1)

        half_width = z * s / np.sqrt(n)

        if half_width <= beta:
            return np.mean(y_vals), n, half_width




def function(x):
    return math.sqrt(2 * math.pi) * max(0, math.fabs(x) - 1.5)

print(MCE(function, norm(loc = 0.0, scale = 1.0), 0.05))
print(MCETest(function, norm(loc = 0.0, scale = 1.0), 0.05))