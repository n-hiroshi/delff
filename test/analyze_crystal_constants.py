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


# delff - original libraries https://github.com/n-hiroshi/delff.git
import delff.lammpsff as lff
import delff.gaussianff as gff
import delff.evalfunc as ef
import delff.util as util
from delff.gaussianhandler import GaussianHandler
from delff.delff_lat import *
from delff.objects import *
from  delff import energy,opt,metaopt,delff_lat

Array = jnp.ndarray
f64 = jnp.float64
i64 = jnp.int64
kcalM2Hartree = 627.70960803
dielectric_constant = 3.0

def main(lat_gjfs,name):
    #print(lat_gjfs)
    #print('abc alpha beta gamma')

    print('%s '%name,end='')
    for lat_gjf in lat_gjfs:
        GH=GaussianHandler()
        lattice = GH.get_system(lat_gjf)[1]
 
        a = lattice[0,:]
        b = lattice[1,:]
        c = lattice[2,:]
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
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res099/%s_lat_optff99.gjf'%(name,name), './data/%s/res083/%s_lat_optff83.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='antcen'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res092/%s_lat_optff92.gjf'%(name,name), './data/%s/res089/%s_lat_optff89.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='benzac'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res097/%s_lat_optff97.gjf'%(name,name), './data/%s/res084/%s_lat_optff84.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='biphen'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res090/%s_lat_optff90.gjf'%(name,name), './data/%s/res082/%s_lat_optff82.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='bpheno'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res098/%s_lat_optff98.gjf'%(name,name), './data/%s/res086/%s_lat_optff86.gjf'%(name,name)]
    main(lat_gjfs,name)

    #name='hunxoe'
    #lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
    #            './data/%s/res070/%s_lat_optff70.gjf'%(name,name), './data/%s/res080/%s_lat_optff80.gjf'%(name,name)]
    #main(lat_gjfs,name)

    name='naphol'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res097/%s_lat_optff97.gjf'%(name,name), './data/%s/res082/%s_lat_optff82.gjf'%(name,name)]
    main(lat_gjfs,name)

    #name='olenic'
    #lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
    #            './data/%s/res070/%s_lat_optff70.gjf'%(name,name), './data/%s/res080/%s_lat_optff80.gjf'%(name,name)]
    #main(lat_gjfs,name)

    name='pencen'
    lat_gjfs = ['./data/%s/%s_lat211.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res090/%s_lat_optff90.gjf'%(name,name), './data/%s/res086/%s_lat_optff86.gjf'%(name,name)]
    main(lat_gjfs,name)

    name='tphben'
    lat_gjfs = ['./data/%s/%s_lat.gjf'%(name,name), './data/%s/res070/%s_lat_iniff70.gjf'%(name,name),\
                './data/%s/res060/%s_lat_optff60.gjf'%(name,name), './data/%s/res070/%s_lat_optff70.gjf'%(name,name),
                './data/%s/res096/%s_lat_optff96.gjf'%(name,name), './data/%s/res082/%s_lat_optff82.gjf'%(name,name)]
    main(lat_gjfs,name)
