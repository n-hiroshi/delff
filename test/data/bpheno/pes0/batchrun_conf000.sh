#!/bin/bash
#PBS -q C
#PBS -N mos_g16
#PBS -l select=1:ncpus=32:mpiprocs=1:mem=120GB
cd $PBS_O_WORKDIR
source /etc/profile.d/modules.sh
module load pgi/15.10
unset LD_LIBRARY_PATH
export GAUSS_SCRDIR=/tmp2/scr
export g16root=/opt/ap/G16/B01
source $g16root/g16/bsd/g16.profile
g16 _conf000.gjf
g16 _conf004.gjf
g16 _conf008.gjf
g16 _conf012.gjf
g16 _conf016.gjf
g16 _conf020.gjf
g16 _conf024.gjf
g16 _conf028.gjf
g16 _conf032.gjf
