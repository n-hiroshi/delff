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
g16 _conf001.gjf
g16 _conf005.gjf
g16 _conf009.gjf
g16 _conf013.gjf
g16 _conf017.gjf
g16 _conf021.gjf
g16 _conf025.gjf
g16 _conf029.gjf
g16 _conf033.gjf
g16 _conf037.gjf
g16 _conf041.gjf
g16 _conf045.gjf
g16 _conf049.gjf
g16 _conf053.gjf
g16 _conf057.gjf
g16 _conf061.gjf
g16 _conf065.gjf
g16 _conf069.gjf
