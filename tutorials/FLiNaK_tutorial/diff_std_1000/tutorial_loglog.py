import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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


for filename in filenames:
    step, temp, density, msdLi, header = read_lammps_log(filename)

    timestep = 1e-15  # ps -> s
    t = (np.array(step) - 30000)

    # Берем только положительные значения
    mask = (t > 0) & (msdLi > 0)
    t_all = t[mask]
    msd_all = np.array(msdLi)[mask]

    # Выбираем линейный участок с конца (например, последние 20% данных)
    N = len(t_all)
    tail_mask = np.arange(N) >= N * 0.3
    t_tail = t_all[tail_mask]
    msd_tail = msd_all[tail_mask]

    # Фит в log-log пространстве
    log_t_tail = np.log(t_tail)
    log_msd_tail = np.log(msd_tail)

    model = LinearRegression()
    model.fit(log_t_tail.reshape(-1,1), log_msd_tail)
    
    # Предсказание по всему диапазону t
    log_msd_pred = model.predict(np.log(t_all).reshape(-1,1))
    msd_pred = np.exp(log_msd_pred)

    # График
    plt.figure(figsize=(8,6))
    plt.loglog(t_all, msd_all, label='MSD_Li(t)')
    plt.loglog(t_all, msd_pred, label=f'Fit log-log (tail), R²={r2_score(log_msd_tail, model.predict(log_t_tail.reshape(-1,1))):.3f}', color='red')
    plt.xlabel('t, с')
    plt.ylabel('MSD_Li, Å²')
    plt.title('MSD_Li vs t (log-log)')
    plt.legend()
    plt.savefig(f'{filename}.png')

    # Наклон линии
    slope = model.coef_[0]
    print(f"{filename}: slope (MSD ~ t^k) = {slope:.3f}")


