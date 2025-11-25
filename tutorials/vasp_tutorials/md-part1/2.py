# plot_with_pymatgen.py
from pathlib import Path
import matplotlib.pyplot as plt

path = Path("./e01_solid-cd-Si") / "vasprun.xml"

if not path.exists():
    raise SystemExit("vasprun.xml не найден в ./e01_solid-cd-Si")

from pymatgen.io.vasp import Vasprun

# парсим vasprun; пропускаем тяжёлые разделы (по желанию)
vr = Vasprun(str(path), parse_dos=False, parse_eigen=False)

# final_energy (последняя общая энергия)
final = vr.final_energy
print("final_energy (vasprun):", final)

# если есть пошаговые (ionic_steps) — пытаемся получить energies
energies = []
if hasattr(vr, "ionic_steps") and vr.ionic_steps:
    # ionic_steps может быть списком словарей; проверяем возможные ключи
    for step in vr.ionic_steps:
        # разные версии pymatgen могут хранить энергию под разными ключами,
        # попробуем несколько вариантов надёжно
        if isinstance(step, dict):
            for key in ("e_fr_energy", "energy", "e_fr_energy_ev", "energy_fb"):
                if key in step:
                    energies.append(float(step[key]))
                    break
            else:
                # если словарь, но нет знакомых ключей — смотрим на значение 'energy'
                val = step.get("energy")
                if val is not None:
                    energies.append(float(val))
        else:
            # если элемент не словарь — пытаемся str->float (крайний случай)
            try:
                energies.append(float(step))
            except Exception:
                pass

# fallback: если пошаговых нет — возьмём final как единственную точку
if not energies:
    energies = [final]

# построение
plt.figure(figsize=(7,4))
plt.plot(range(len(energies)), energies, marker='o', linestyle='-')
plt.xlabel("Step")
plt.ylabel("Energy (eV)")
plt.title("Energy from vasprun.xml (pymatgen)")
plt.grid(True)
out = Path("./e01_solid-cd-Si") / "energy_pymatgen.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
print("Сохранено:", out)
