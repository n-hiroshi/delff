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

# mos - original libraries http://43.25.153.117:4003/nakano/mos
from mos.infra import System, MolType, Mol
from mos.ext.gaussian import GaussianCartHandler

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
    refdir = "./data/blue63open/"
    root = refdir + "res%03d/"%idx
    os.makedirs(root,exist_ok=True)
    name = "blue63open"

    taskids = [0,2,3,4,5,6,7,8]
    task_balance = [1,1,1,1,1,1,1,1]
    #taskids = [2,3,4,5,6,7,8]
    #task_balance = [1,1,1,1,1,1,1]
    #taskids = [0]
    #task_balance = [1]
    #taskids = [0,1]
    #task_balance = [1,[0.1,0.1]]

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
    in_settings_file = refdir + 'blue63open.in.settings'
    data_file = refdir + 'blue63open.data'
    monomer_gjf = refdir + 'blue63open.gjf'
    atomtypesets_from1 = [[1],[2],[3],[4],[5],[6],[7],[8],[9],
                          [10],[11],[12],[13],[14],[15],[16],[17],[18],[19],
                          [20],[21],[22],[23],[24],[25],[26],[27],[28],[29],
                          [30],[31,32,33],[34],[35],[36],[37],[38],[39],
                          [40],[41],[42],[43,44],[45],[46,47],[48],[49,50],
                          [50],[51],[52],[53],[54],[55],[56,57,58],[59],[60],
                          [61],[62,63,64],[65],[66],[67]]
    atomtypesets = [[v-1 for v in list0] for list0 in atomtypesets_from1]

    # crystal lat
    lat_gjf  = refdir + 'dummy_blue63open_crystal.gjf'

    Hsubl    = 100 #[kJ/mol] at 298.15K ref: McDonagh JCIM56(2016)2162
    T = 298.15
    R = 8.314462618 #[J/mol/K] K=Kelvin

    Hsubl   *= 0.00038087983241287375 # kJ/mol to Hartee
    R       *=  1/1000 * 0.00038087983241287375 # J/mol/K to kJ/mol/K to Hartee/K
    print('Hsubl: %8.5e [Hartree]'%Hsubl)
    Ulattice = - Hsubl - 2*R*T
    print('Ulattice: %8.5e [Hartree]'%Ulattice)

    # pes
    pes_dirs = [refdir + 'pes0/',refdir + 'pes1/',refdir + 'pes2/',refdir + 'pes3/',
                refdir + 'pes4/',refdir + 'pes5/',refdir + 'pes6/']

    # ff_load
    path_ff_load = ''
    #path_ff_load = refdir + 'ff_opt07.pickle'

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
                                     total_charge = 1.0 ################# CHARGE = 1
                                     )
    if algo in ['BO']:
        ff_opt = delff_lat.run(ff_ini,work,algo,maxiter=1000,pes_dirs=pes_dirs)
    elif algo in ['FD','SD']:
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
