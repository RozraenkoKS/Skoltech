set term png
set output "loop_energy.png"

set title "Convergence study beta-tin Si"
set xlabel "no of k points"
set ylabel "Total energy (eV)"

plot "loop.dat" using 1:4 w lp

set output "loop_lattice.png"
set ylabel "length of first lattice vector (Angstrom)"

plot "loop.dat" using 1:(4.8*$10) w lp
