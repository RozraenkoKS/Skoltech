#!/bin/bash   
#SBATCH -J tutorial #имя в очереди
##SBATCH -p high #имя партиции, high, MMM, bigmem
#SBATCH -N 1 #количество узлов, оптимально 1
#SBATCH -n 16 #число ядер, оптимально 16 node-mmm[01-14], 36 node-mmm[15-22]
#SBATCH -o %x.e%j #tutorial.eID
#SBATCH -t 03:00:00 #время выпонения, для снятия ограничения: 
##SBATCH --time=UNLIMITED

module purge
module load Compiler/Intel/18u4
ulimit -s unlimited

mpirun -np 16 ~/resources/bin/vasp_std
