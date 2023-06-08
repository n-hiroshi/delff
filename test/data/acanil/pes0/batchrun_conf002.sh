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
g16 _conf002.gjf
g16 _conf006.gjf
g16 _conf010.gjf
g16 _conf014.gjf
g16 _conf018.gjf
g16 _conf022.gjf
g16 _conf026.gjf
g16 _conf030.gjf
g16 _conf034.gjf
g16 _conf038.gjf
g16 _conf042.gjf
g16 _conf046.gjf
g16 _conf050.gjf
g16 _conf054.gjf
g16 _conf058.gjf
g16 _conf062.gjf
g16 _conf066.gjf
g16 _conf070.gjf
