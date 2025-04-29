import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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

filenames = ['log1600_1400.lammps', 'log1400_1200.lammps', 'log1200_1000.lammps',
             'log1000_800.lammps', 'log800_600.lammps']
temps = [1400, 1200, 1000, 800, 600]

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

    y_pred = model.predict(X)

    plt.figure()
    plt.scatter(X, y, label='Данные', color='b', alpha=0.6)
    plt.plot(X, y_pred, label=f'Лин. регрессия: y = {k:.4f}·x + {b:.4f}', color='r')
    plt.xlabel('Время, с')
    plt.ylabel('Среднеквадратичное отклонение, cм²')
    plt.legend()
    plt.title(f'Зависимость среднеквадратичного отклонения от времени при температуре T = {temps[i]}')
    plt.tight_layout()
    i += 1
plt.show()

X = np.array(temps).reshape(-1, 1)   
y = np.array(densities)             

model = LinearRegression()
model.fit(X, y)

k = model.coef_[0]
b = model.intercept_

y_pred = model.predict(X)

plt.figure()
plt.scatter(X, y, label='Данные', color='b', alpha=0.6)
plt.plot(X, y_pred, label=f'Лин. регрессия: y = {k:.4f}·x + {b:.4f}', color='r')
plt.xlabel('Температура, К')
plt.ylabel('Плотность, г/см³')
plt.legend()
plt.title('Зависимость плотности от температуры')
plt.tight_layout()
plt.show()

y = np.array(diffs)             

plt.figure()
plt.scatter(X, y, label='Данные', color='b', alpha=0.6)
plt.xlabel('Температура, К')
plt.ylabel('Коэффициент диффузии, см²/с')
plt.legend()
plt.title('Зависимость коэффициента диффузии от температуры')
plt.tight_layout()
plt.show()

        