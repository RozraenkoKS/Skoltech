from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter
import matplotlib.pyplot as plt  # для работы с figure

# Загружаем vasprun.xml
vasp_run = Vasprun("vasprun.xml", parse_projected_eigen=True)

# Получаем band structure
bs = vasp_run.get_band_structure(line_mode=True)

# Создаём plotter
plotter = BSPlotter(bs)

# Получаем axes
ax = plotter.get_plot()

# Показываем график через matplotlib
plt.savefig('e03.png')
plt.show()  # здесь plt.show(), а не ax.show()


