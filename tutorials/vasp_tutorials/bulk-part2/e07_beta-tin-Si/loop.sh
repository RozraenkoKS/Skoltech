for k in  03 04 05 06 07 08 09 10 11 12 13
do
cp POSCAR.bk POSCAR 
vasp_rm

cat > KPOINTS << EOF
K-Points
 0
Monkhorst Pack
$k $k $k
0  0  0
EOF

mpirun -np 2 vasp_std
cp CONTCAR POSCAR
mpirun -np 2 vasp_std

cp CONTCAR POSCAR.k$k

v=$(awk 'NR==3' CONTCAR)
en=$(tail -n 1 OSZICAR)
echo $k $en $v  >> loop.dat
done