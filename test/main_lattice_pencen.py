import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from jax import numpy as jnp, lax, vmap, jacfwd, grad

# delff - original libraries https://github.com/n-hiroshi/delff.git
import delff.lammpsff as lff
import delff.gaussianff as gff
import delff.evalfunc as ef
import delff.util as util
from delff.objects import *
from  delff import energy,opt,metaopt,delff_lat

Array = jnp.ndarray
f64 = jnp.float64
i64 = jnp.int64
kcalM2Hartree = 627.50960803

def main(idx,algo):
    refdir = "./data/pencen/"
    root = refdir + "res%03d/"%idx
    os.makedirs(root,exist_ok=True)
    name = "pencen"

    #taskids = [0,1,2]
    #task_balance = [1,[1,1],1]
    #taskids = [0]
    #task_balance = [1]
    taskids = [0,1]
    task_balance = [1,[1,1]]

    assert taskids[0] == 0
    assert len(task_balance) == len(taskids)

    print('# settings')
    print('algo: %s'%algo)
    print('root: %s'%root)
    print('name: %s'%name)
    print('idx: %02d'%idx)
    print('taskids: ',taskids)
    print('task_balance: ',task_balance)

    # monomer
    in_settings_file = refdir + 'pencen.in.settings'
    data_file = refdir + 'pencen.data'
    monomer_gjf = refdir + 'pencen.gjf'
    atomtypesets_from1 = [[2,7],[1,3,4,9],[5,6,8,13],[10,11,12,16],[14,15,17,20],[18,19,21,22],[23,26],[24,25,27,28],[29,30,31,34],[32,33,35,36]]
    atomtypesets = [[v-1 for v in list0] for list0 in atomtypesets_from1]

    # crystal lat
    lat_gjf  = refdir + 'pencen_lat211.gjf'

    Hsubl    = 117.3 #[kJ/mol] at 298.15K ref: Shalevc OrganicElect 14(2013)94
    T = 523.15 # 250 + 273.15
    R = 8.314462618 #[J/mol/K] K=Kelvin

    Hsubl   *= 0.00038087983241287375 # kJ/mol to Hartee
    R       *=  1/1000 * 0.00038087983241287375 # J/mol/K to kJ/mol/K to Hartee/K
    print('Hsubl: %8.5e [Hartree]'%Hsubl)
    Ulattice = - Hsubl - 2*R*T
    print('Ulattice: %8.5e [Hartree]'%Ulattice)

    # pes
    #pes_dirs = [refdir + 'pes0/']
    pes_dirs = []

    # ff_load
    path_ff_load = ''
    #path_ff_load = root + 'ff_opt09.pickle'

    #    
    ff_ini, work = delff_lat.prepare(root,
                                     name,
                                     idx,
                                     task_balance,
                                     taskids,
                                     monomer_gjf,
                                     atomtypesets,
                                     in_settings_file,
                                     data_file,
                                     path_ff_load = path_ff_load,
                                     lat_gjf = lat_gjf,
                                     Ulattice = Ulattice,
                                     pes_dirs = pes_dirs,
                                     dielectric_constant = 3.0,
                                     ewald=True,
                                     alpha_ewald= 0.18659215,
                                     nkmaxs=jnp.array([1,1,3])
                                     )
    ff_opt = delff_lat.run(ff_ini,work,algo,maxiter=100,pes_dirs=pes_dirs)


if __name__ == '__main__':
    idx = int(sys.argv[1])
    algo = sys.argv[2]
    main(idx,algo)
