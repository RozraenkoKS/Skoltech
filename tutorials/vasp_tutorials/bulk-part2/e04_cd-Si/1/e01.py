import matplotlib.pyplot as plt
import numpy as np

# читаем данные
a = []
E = []

with open("loop_lattice_constant.dat") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4:
            a.append(float(parts[0]))
            E.append(float(parts[3]))

a = np.array(a)
E = np.array(E)

# ищем минимум
idx_min = np.argmin(E)
a_min = a[idx_min]
E_min = E[idx_min]

# строим график
plt.figure(figsize=(7,5))
plt.plot(a, E, "o-", color="navy", label="VASP data")
plt.scatter(a_min, E_min, color="red", zorder=5, label=f"min a {a_min:.2f} Å")

# оформление
plt.title("Equation of State: fcc Si", fontsize=14)
plt.xlabel("Lattice constant a [Å]", fontsize=12)
plt.ylabel("Total free energy [eV]", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('loop_lattice_constant.png')

plt.show()

