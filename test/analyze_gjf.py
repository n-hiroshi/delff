import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append('/home/nakano/delff/')
sys.path.append('/home/nakano/mos/')
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from jax import numpy as jnp, lax, vmap, jacfwd, grad

# mos - original libraries http://43.25.153.117:4603/nakano/mos
from mos.infra import System, MolType, Mol
from mos.ext.gaussian import GaussianCartHandler

# delff - original libraries https://github.com/n-hiroshi/delff.git
import delff.lammpsff as lff
import delff.gaussianff as gff
import delff.evalfunc as ef
import delff.util as util
from delff.delff_lat import *
from delff.objects import *
from  delff import energy,opt,metaopt,delff_lat

Array = jnp.ndarray
f64 = jnp.float64
i64 = jnp.int64
kcalM2Hartree = 627.56960803
dielectric_constant = 3.0

def main(lat_gjfs,name):
    #print(lat_gjfs)
    #print('abc alpha beta gamma')

    print('%s '%name,end='')
    for lat_gjf in lat_gjfs:
        sys_tgt_lat = gff.get_sys_from_gjf(lat_gjf)
 
        a = sys_tgt_lat.lattice[0,:]
        b = sys_tgt_lat.lattice[1,:]
        c = sys_tgt_lat.lattice[2,:]
        norma = jnp.linalg.norm(a)
        normb = jnp.linalg.norm(b)
        normc = jnp.linalg.norm(c)
 
        cosalpha = jnp.dot(b/normb,c/normc)
        cosbeta  = jnp.dot(c/normc,a/norma)
        cosgamma = jnp.dot(a/norma,b/normb)
 
        alpha = 180/np.pi*jnp.arccos(cosalpha)
        beta  = 180/np.pi*jnp.arccos(cosbeta)
        gamma = 180/np.pi*jnp.arccos(cosgamma)
 
        print('%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f '%(norma,normb,normc,alpha,beta,gamma),end='')

    print()


if __name__ == '__main__':
    #lat_gjfs = #sys.argv[1:5]
    
    name='acanil'#sys.argv[1]
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='antcen'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='benzac'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='biphen'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='bpheno'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='hunxoe'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='naphol'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='olenic'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='pencen'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='tphben'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res056/%s_lat_iniff56.gjf'%(name,name),\
                './data/%s/res056/%s_lat_optff56.gjf'%(name,name), './data/%s/res046/%s_lat_optff46.gjf'%(name,name)]
    main(lat_gjfs,name)
