from pymatgen.io.vasp import Vasprun
import matplotlib.pyplot as plt

vasp = Vasprun("vasprun.xml", parse_projected_eigen=True)
dos = vasp.complete_dos

energies = dos.energies - dos.efermi          # энергия относительно Fermi
tdos = dos.get_densities()                    # total DOS

plt.plot(energies, tdos, color='blue', lw=1.5)
plt.axvline(0.0, color='k', linestyle='--', lw=0.8)  # Fermi level
plt.xlabel("E - E_F (eV)")
plt.ylabel("DOS (states/eV)")
plt.title("Density of States for fcc Si")
plt.grid(True)
plt.savefig('e02.png')
plt.show()

