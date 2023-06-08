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
    refdir = "./data/benzac/"
    root = refdir + "res%03d/"%idx
    os.makedirs(root,exist_ok=True)
    name = "benzac"

    taskids = [0,1,2,3]
    task_balance = [1,[1,1],1,1]
    #taskids = [0,2,3]
    #task_balance = [1,1,1]

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
    in_settings_file = refdir + 'benzac.in.settings'
    data_file = refdir + 'benzac.data'
    monomer_gjf = refdir + 'benzac.gjf'
    atomtypesets_from1 = [[1],[2,3],[4,7],[9],[5,8],[10,13],[14],[6],[11],[12],[15]]
    atomtypesets = [[v-1 for v in list0] for list0 in atomtypesets_from1]

    # crystal lat
    lat_gjf  = refdir + 'benzac_lat211.gjf'

    Hsubl    = 89.2 # [KJ/mol]  McDonagh2016
    T = 298.15 # K
    R = 8.314462618 #[J/mol/K] K=Kelvin

    Hsubl   *= 0.00038087983241287375 # kJ/mol to Hartee
    R       *=  1/1000 * 0.00038087983241287375 # J/mol/K to kJ/mol/K to Hartee/K
    print('Hsubl: %8.5e [Hartree]'%Hsubl)
    Ulattice = - Hsubl - 2*R*T 
    # Ulatticeは1分子ごとの値。UnitCellに複数分子がある場合は対応するUlatt_ffを
    #その分子数で割る必要がある。-> evalfunc.pyに実装済み。
    print('Ulattice: %8.5e [Hartree]'%Ulattice)

    # pes
    pes_dirs = [refdir + 'pes0/',refdir + 'pes1/']
    #pes_dirs = []

    # ff_load
    path_ff_load = ''
    #path_ff_load = root + 'ff_opt08.pickle'

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
                                     param_balance = jnp.asarray([1,1,1,0,1,0]),
                                     path_ff_load = path_ff_load,
                                     lat_gjf = lat_gjf,
                                     Ulattice = Ulattice,
                                     pes_dirs = pes_dirs,
                                     dielectric_constant = 3.0,
                                     ewald=True,
                                     alpha_ewald= 0.20623066,
                                     nkmaxs=jnp.array([2,1,3])
                                     )
    if algo in ['BO']:
        ff_opt = delff_lat.run(ff_ini,work,algo,maxiter=1000,pes_dirs=pes_dirs)
    elif algo in ['FDM','SDM']:
        ff_opt = delff_lat.run(ff_ini,work,algo,maxiter=100,pes_dirs=pes_dirs)
    elif algo in ['FDSD']:
        ff_opt = delff_lat.run(ff_ini,work,'FD',maxiter=50,pes_dirs=pes_dirs)
        ff_opt = delff_lat.run(ff_opt,work,'SD',maxiter=50,pes_dirs=pes_dirs)
    elif algo in ['BOSD']:
        ff_opt = delff_lat.run(ff_ini,work,'BO',maxiter=500,pes_dirs=pes_dirs)
        ff_opt = delff_lat.run(ff_opt,work,'SD',maxiter=50,pes_dirs=pes_dirs)


if __name__ == '__main__':
    idx = int(sys.argv[1])
    algo = sys.argv[2]
    main(idx,algo)
