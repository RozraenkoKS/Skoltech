from pymatgen.core import Structure

my_struc = Structure.from_file("./e01_solid-cd-Si/Si_mp-149_conventional_standard.cif")

# make a 2x2x2 supercell
my_struc.make_supercell(2)

# write supercell to POSCAR format with specified filename
my_struc.to(fmt="poscar", filename="./e01_solid-cd-Si/POSCAR")

# A POSCAR file will appear at $TUTORIALS/MD/e01_solid-cd-Si
# You may need to refresh the file browser to see it