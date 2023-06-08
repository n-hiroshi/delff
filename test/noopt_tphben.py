import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append('/home/nakano/mos/')
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
    refdir = "./data/tphben/"
    root = refdir + "res%03d/"%idx
    os.makedirs(root,exist_ok=True)
    name = "tphben"

    taskids = [0,1,2]
    task_balance = [1,[1,1],1]
    #taskids = [0]
    #task_balance = [1]
    #taskids = [0,1]
    #task_balance = [1,[1,1]]

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
    in_settings_file = refdir + 'tphben.in.settings'
    data_file = refdir + 'tphben.data'
    monomer_gjf = refdir + 'tphben.gjf'
    atomtypesets_from1 = [[1,4,7],[2,3,9],[5,8,14],[6,10,13],[11,12,15,16,21,22],[17,19,23,25,30,32],[27,34,38],[18,20,24,26,31,33],[28,29,35,36,39,40],[37,41,42]]
    atomtypesets = [[v-1 for v in list0] for list0 in atomtypesets_from1]

    # crystal lat
    lat_gjf  = refdir + 'tphben_lat.gjf'

    Hsubl    = 147.8 # [KJ/mol]  McDonagh2016
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
    pes_dirs = [refdir + 'pes0/']
    #pes_dirs = []

    # ff_load
    #path_ff_load = ''
    path_ff_load = root + 'ff_opt%02d.pickle'%idx

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
                                     alpha_ewald= 0.14572648,
                                     nkmaxs=jnp.array([1,2,1])
                                     )
    if algo in ['BO']:
        ff_opt = delff_lat.noopt(ff_ini,work,algo)
    elif algo in ['FDM','SDM']:
        ff_opt = delff_lat.noopt(ff_ini,work,algo)
    elif algo in ['FDSD']:
        ff_opt = delff_lat.noopt(ff_ini,work,'FD')
        ff_opt = delff_lat.noopt(ff_opt,work,'SD')
    elif algo in ['BOSD']:
        ff_opt = delff_lat.noopt(ff_ini,work,'BO')
        ff_opt = delff_lat.noopt(ff_opt,work,'SD')


if __name__ == '__main__':
    idx = int(sys.argv[1])
    algo = sys.argv[2]
    main(idx,algo)
