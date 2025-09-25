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

filenames = [

]

diffs = []
N = np.array([93, 744, 2511, 5952])

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

plt.plot(np.power(N, -1/3), diffs)
plt.title('Зависимость коэффиценту диффузии от концентрации')
plt.xlabel('N^(-1/3), отн. ед.')
plt.ylabel('D, см²/с')
plt.show()

        
