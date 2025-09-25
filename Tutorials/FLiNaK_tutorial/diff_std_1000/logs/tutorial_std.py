"""Рисует гистограмму получившихся в модуляциях диффузий, выводит mean, std.

На входе нужно указать используемые filenames
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns

def read_lammps_log(filename):
    step = []
    temp = []
    density = []
    msdLi = []

    with open(filename, 'r') as f:
        line = f.readline().strip()
        while not line.startswith("Step"):
            line = f.readline().strip()

        line = f.readline().strip()
        while not line.startswith("Step"):
            line = f.readline().strip()
        
        header = line.split()

        words = f.readline().strip().split()
        while words and words[0] != "Loop":
            step.append(float(words[0]))
            temp.append(float(words[4]))
            density.append(float(words[8]))
            msdLi.append(float(words[9]))
            words = f.readline().strip().split()

    return np.array(step), temp, density, np.array(msdLi), header

filenames = [
    "tutorial.e4776856",
    "tutorial.e4776858",
    "tutorial.e4776864",
    "tutorial.e4776866",
    "tutorial.e4776867",
    "tutorial.e4776869",
    "tutorial.e4776870",
    "tutorial.e4776871",
    "tutorial.e4776872",
    "tutorial.e4776873",
]

temps = [1000] * 10

densities = []
diffs = []

i = 0
for filename in filenames:
    step, temp, density, msdLi, header = read_lammps_log(filename)

    densities.append(np.mean(density))
    
    X = ((step - 30000) * 10 ** (-15)).reshape(-1, 1)   
    y = msdLi * 10 ** (-16)          

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    k = model.coef_[0]
    b = model.intercept_
    diffs.append(k / 6)

y = np.array(diffs)       

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6,4))
sns.histplot(y, kde=True, color="skyblue", edgecolor="black", bins=20)
plt.xlabel("Значение", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.title("Гистограмма выборки", fontsize=14)
plt.show()

print(f"Среднее: {np.mean(y):.3e}")
print(f"Стандартное отклонение: {np.std(y):.3e}")
