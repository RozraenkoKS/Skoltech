from pymatgen.io.vasp import Vasprun
import matplotlib.pyplot as plt

vasp = Vasprun("vasprun4.xml", parse_projected_eigen=True)
dos = vasp.complete_dos

energies = dos.energies - dos.efermi  # Fermi = 0
tdos = dos.get_densities()            # total DOS

plt.plot(energies, tdos)
plt.axvline(0, color='k', lw=0.8)    # Fermi
plt.savefig('vasprun4.png')
plt.show()