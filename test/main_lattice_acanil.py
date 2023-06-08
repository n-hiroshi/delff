import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#sys.path.append('/home/nakano/mos/')
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from jax import numpy as jnp, lax, vmap, jacfwd, grad

# mos - original libraries http://43.25.153.117:4003/nakano/mos
#from mos.infra import System, MolType, Mol
#from mos.ext.gaussian import GaussianCartHandler

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
    
    print('\n\n\n############### SETTINGS ###############')
    refdir = "./data/acanil/"
    root = refdir + "res%03d/"%idx
    os.makedirs(root,exist_ok=True)
    name = "acanil"

    taskids = [0,1,2,3]
    task_balance = [1,[1,1],1,1]
    #taskids = [0,2,3]
    #task_balance = [1,1,1]

    assert taskids[0] == 0
    assert len(task_balance) == len(taskids)

    print('# algo: %s'%algo)
    print('# root: %s'%root)
    print('# name: %s'%name)
    print('# idx: %02d'%idx)
    print('# taskids: ',taskids)
    print('# task_balance: ',task_balance)

    # monomer
    in_settings_file = refdir + 'acanil.in.settings'
    data_file = refdir + 'acanil.data'
    monomer_gjf = refdir + 'acanil.gjf'
    atomtypesets_from1 = [[1],[2,3],[4,7],[9],[5,8],[10,13],[14],[6],[12],[11],[16],[15],[17,18,19]]
    print('# atomtypesets(start from 1)',atomtypesets_from1)
    atomtypesets = [[v-1 for v in list0] for list0 in atomtypesets_from1]

    # crystal lat
    lat_gjf  = refdir + 'acanil_lat.gjf'

    Hsubl    = 99.8 # [KJ/mol]  McDonagh2016
    T = 298 # K
    R = 8.314462618 #[J/mol/K] K=Kelvin

    Hsubl   *= 0.00038087983241287375 # kJ/mol to Hartee
    R       *=  1/1000 * 0.00038087983241287375 # J/mol/K to kJ/mol/K to Hartee/K
    print('# Hsubl: %8.5e [Hartree]'%Hsubl)
    Ulattice = - Hsubl - 2*R*T 
    print('# Ulattice: %8.5e [Hartree]'%Ulattice)

    # pes
    pes_dirs = [refdir + 'pes0/',refdir + 'pes1/']
    print('# pes data:',pes_dirs)

    # ff_load
    path_ff_load = ''

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
                                     alpha_ewald= 0.21551273,
                                     nkmaxs=jnp.array([3,2,2])
                                     )
    ff_opt = delff_lat.run(ff_ini,work,algo,maxiter=100,pes_dirs=pes_dirs)


if __name__ == '__main__':
    idx = int(sys.argv[1])
    algo = sys.argv[2]
    main(idx,algo)
