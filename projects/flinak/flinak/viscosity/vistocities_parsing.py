import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.optimize import curve_fit

# -----------------------------
# 1. Список логов
# -----------------------------
logfiles = [
    'viscosityFlinak.e4857869',
    'viscosityFlinak.e4857881',
    'viscosityFlinak.e4857888',
    'viscosityFlinak.e4857921'
]

# -----------------------------
# 2. Параметры
# -----------------------------
dt = 0.001    # timestep в ps

# -----------------------------
# 3. Функция для парсинга одного лога
# -----------------------------
def parse_log(logfile):
    steps, v11, v22, v33 = [], [], [], []
    read_data = False
    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Step') and 'v_v11' in line:
                read_data = True
                continue
            if read_data:
                if line == '' or line.startswith('Loop') or line.startswith('Performance'):
                    read_data = False
                    continue
                data = line.split()
                if len(data) >= 9:
                    try:
                        steps.append(int(data[0]))
                        v11.append(float(data[6]))
                        v22.append(float(data[7]))
                        v33.append(float(data[8]))
                    except ValueError:
                        continue
    steps = np.array(steps)
    v_mean = (np.array(v11) + np.array(v22) + np.array(v33))/3.0
    time_ps = steps * dt
    return time_ps, v_mean

# -----------------------------
# 4. Фитинг двойной экспонентой
# -----------------------------
def double_exp(t, A, alpha, tau1, tau2):
    return A*alpha*tau1*(1-np.exp(-t/tau1)) + A*(1-alpha)*tau2*(1-np.exp(-t/tau2))

p0 = [0.004, 0.5, 1e4, 1e5]
bounds = ([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf])

# -----------------------------
# 5. Обработка всех файлов
# -----------------------------
plt.figure(figsize=(10,6))
etas = []

i = 0
for logfile in logfiles:
    i += 1
    time_ps, v_mean = parse_log(logfile)
    popt, _ = curve_fit(double_exp, time_ps, v_mean, p0=p0, bounds=bounds)
    A_fit, alpha_fit, tau1_fit, tau2_fit = popt
    eta_inf = A_fit*alpha_fit*tau1_fit + A_fit*(1-alpha_fit)*tau2_fit
    etas.append(eta_inf)

    print(f"Файл: {logfile}")
    print(f"  A = {A_fit:.6g}, alpha = {alpha_fit:.6g}, tau1 = {tau1_fit:.6g} ps, tau2 = {tau2_fit:.6g} ps")
    print(f"  Оценка вязкости η∞ = {eta_inf:.6g} Pa.s\n")

    plt.plot(time_ps, v_mean, label=f'{logfile} (данные)', alpha=0.5, c=f'C{i}')
    plt.plot(time_ps, double_exp(time_ps, *popt), '-', label=f'{logfile} (фитинг)', c=f'C{i}')

plt.xlabel('Time [ps]')
plt.ylabel('Вязкость [Pa.s]')
plt.title('Сходимость вязкости и фитинг для нескольких логов')
plt.legend()
plt.grid(True)
plt.show()

alpha = 0.95
n = len(etas)
etas = np.array(etas)
x = np.mean(etas)
t = sps.t(n - 1).ppf((1 + alpha) / 2)
s = np.std(etas)
left = x - t * s / np.sqrt(n - 1)
right = x + t * s / np.sqrt(n - 1)

print(f'Доверительный интервал для η∞ = {x:.6g} уровня доверия α=0.95: ({left:.6g}, {right:.6g})')

