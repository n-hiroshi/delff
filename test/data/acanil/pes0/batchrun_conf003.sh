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
g16 _conf003.gjf
g16 _conf007.gjf
g16 _conf011.gjf
g16 _conf015.gjf
g16 _conf019.gjf
g16 _conf023.gjf
g16 _conf027.gjf
g16 _conf031.gjf
g16 _conf035.gjf
g16 _conf039.gjf
g16 _conf043.gjf
g16 _conf047.gjf
g16 _conf051.gjf
g16 _conf055.gjf
g16 _conf059.gjf
g16 _conf063.gjf
g16 _conf067.gjf
g16 _conf071.gjf
