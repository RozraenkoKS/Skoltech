import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# 1. Парсинг лога LAMMPS
# -----------------------------
logfile = 'viscosityFlinak.e4857869'  # твой лог LAMMPS

steps = []
v11 = []
v22 = []
v33 = []

read_data = False

with open(logfile, 'r') as f:
    for line in f:
        line = line.strip()
        # Начало блока с колонками Step … v_v11
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

# Преобразуем в numpy
steps = np.array(steps)
v11 = np.array(v11)
v22 = np.array(v22)
v33 = np.array(v33)

# Среднее по трём компонентам
v_mean = (v11 + v22 + v33)/3.0

# -----------------------------
# 2. Корректное время
# -----------------------------
dt = 0.001    # timestep в ps
p = 20000     # correlation length, как в твоем LAMMPS скрипте
s = 1         # sample interval

# Время для подгонки (в пс)
time_ps = steps * dt   # т.к. шаги уже учтены в колонке Step
# Если хочешь в секундах: time_s = time_ps * 1e-12

# -----------------------------
# 3. Фитинг двойной экспонентой
# -----------------------------
def double_exp(t, A, alpha, tau1, tau2):
    return A*alpha*tau1*(1-np.exp(-t/tau1)) + A*(1-alpha)*tau2*(1-np.exp(-t/tau2))

# Начальные приближения: η ~ 0.004, alpha ~ 0.5, τ1, τ2 ~ 1e4–1e5 пс
p0 = [0.004, 0.5, 1e4, 1e5]
bounds = ([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf])  # положительные времена, alpha в [0,1]

popt, pcov = curve_fit(double_exp, time_ps, v_mean, p0=p0, bounds=bounds)

A_fit, alpha_fit, tau1_fit, tau2_fit = popt
eta_inf = A_fit*alpha_fit*tau1_fit + A_fit*(1-alpha_fit)*tau2_fit

print("Параметры подгонки:")
print(f"A = {A_fit:.6g}, alpha = {alpha_fit:.6g}, tau1 = {tau1_fit:.6g} ps, tau2 = {tau2_fit:.6g} ps")
print(f"Оценка вязкости η∞ = {eta_inf:.6g} Pa.s")

# -----------------------------
# 4. Построение графика
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(time_ps, v_mean, 'o', label='Средняя η(t)')
plt.plot(time_ps, double_exp(time_ps, *popt), '-', label='Фитинг двойной экспонентой')
plt.xlabel('Time [ps]')
plt.ylabel('Вязкость [Pa.s]')
plt.title('Сходимость вязкости и фитинг')
plt.legend()
plt.grid(True)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # -----------------------------
# # 1. Парсинг лога LAMMPS
# # -----------------------------
# logfile = 'viscosityFlinak.e4857869'  # путь к твоему логу

# steps = []
# v11 = []
# v22 = []
# v33 = []

# read_data = False

# with open(logfile, 'r') as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith('Step') and 'v_v11' in line:
#             read_data = True
#             continue
#         if read_data:
#             if line == '' or line.startswith('Loop') or line.startswith('Performance'):
#                 read_data = False
#                 continue
#             data = line.split()
#             if len(data) >= 9:
#                 try:
#                     steps.append(int(data[0]))
#                     v11.append(float(data[6]))
#                     v22.append(float(data[7]))
#                     v33.append(float(data[8]))
#                 except ValueError:
#                     continue

# steps = np.array(steps)
# v11 = np.array(v11)
# v22 = np.array(v22)
# v33 = np.array(v33)

# # Среднее по трём компонентам
# v_mean = (v11 + v22 + v33)/3.0

# # -----------------------------
# # 2. Корректное время
# # -----------------------------
# dt = 0.001  # timestep в ps
# time_ps = steps * dt  # шаги уже из колонки Step

# # -----------------------------
# # 3. Фитинг двойной экспонентой
# # -----------------------------
# def double_exp(t, A, alpha, tau1, tau2):
#     return A*alpha*tau1*(1-np.exp(-t/tau1)) + A*(1-alpha)*tau2*(1-np.exp(-t/tau2))

# p0 = [0.004, 0.5, 1e4, 1e5]
# bounds = ([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf])

# popt, pcov = curve_fit(double_exp, time_ps, v_mean, p0=p0, bounds=bounds)
# A_fit, alpha_fit, tau1_fit, tau2_fit = popt
# eta_inf = A_fit*alpha_fit*tau1_fit + A_fit*(1-alpha_fit)*tau2_fit

# print("Параметры подгонки:")
# print(f"A = {A_fit:.6g}, alpha = {alpha_fit:.6g}, tau1 = {tau1_fit:.6g} ps, tau2 = {tau2_fit:.6g} ps")
# print(f"Оценка вязкости η∞ = {eta_inf:.6g} Pa.s")

# # -----------------------------
# # 4. Построение графиков
# # -----------------------------
# plt.figure(figsize=(10,6))
# plt.plot(time_ps, v11, label='v11 (pxy)', alpha=0.7)
# plt.plot(time_ps, v22, label='v22 (pxz)', alpha=0.7)
# plt.plot(time_ps, v33, label='v33 (pyz)', alpha=0.7)
# plt.plot(time_ps, v_mean, 'k--', label='Среднее η(t)')
# plt.plot(time_ps, double_exp(time_ps, *popt), 'r-', label='Fit двойной экспонентой')

# plt.xlabel('Time [ps]')
# plt.ylabel('Viscosity [Pa.s]')
# plt.title('Running integral (автокорреляторы давления)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
