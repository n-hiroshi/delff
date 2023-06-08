import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.append('/home/nakano/mos/')
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from jax import numpy as jnp, lax, vmap, jacfwd, grad, nn as jnn, jit

import delff.evalfunc as ef
import delff.util as util
from delff.objects import *
from  delff import energy,opt,opt_lat,metaopt,rtp,util

kcalM2Hartree = 627.50960803

def define_ewald_params(sys_: System, ffa_: ForceFieldAssignments) -> ForceFieldAssignments:
    """
    This function defines Ewald parameters for a given system using the force field assignments
    and returns the updated force field assignments.

    Arguments:
        sys_ (System): A System object representing the system to calculate Ewald parameters for.
        ffa_ (ForceFieldAssignments): A ForceFieldAssignments object containing force field assignments.

    Returns:
        ffa_ (ForceFieldAssignments): An updated ForceFieldAssignments object with the calculated Ewald parameters.
    """
    alpha_ewald =ffa_.alpha_ewald
    sigma_ewald = 1/alpha_ewald/jnp.sqrt(2.0)
    ffa_ = update(ffa_,alpha_ewald=alpha_ewald,sigma_ewald=sigma_ewald)
    nkvecs = calc_nkvecs(sys_,ffa_)
    ffa_ = update(ffa_,nkvecs=nkvecs)
    return ffa_

def calc_nkvecs(sys_: System, ffa_: ForceFieldAssignments) -> Array:
    """
    This function calculates the k vectors for Ewald summation 
    using combinatorial method and returns the result.

    Arguments:
        sys_ (System): A System object representing the system to calculate k vectors for.
        ffa_ (ForceFieldAssignments): A ForceFieldAssignments 
        object containing force field assignments including cutoff radius and alpha.
    Returns:
        nkvecs (Array): A numpy array containing the k vectors for Ewald summation.
    """
    rcut = ffa_.ccutoff
    Ls = jnp.linalg.norm(sys_.lattice,axis=1)
    alpha = ffa_.alpha_ewald
    print('# Lattice constants of the unit cell:', Ls)
    nkmaxs = ffa_.nkmaxs
    print('## Ewald sum. parameters')
    print('# alpha:', ffa_.alpha_ewald)
    print('# Rcut:', ffa_.ccutoff)
    print('# nkmaxs:', nkmaxs)
    nkvecs = calc_kvecs_combinatorial(sys_,nkmaxs)
    print('# num of kvecs:', nkvecs.shape[0])
    return nkvecs

def calc_kvecs_combinatorial(sys_: System, nkmaxs) -> Array:
    """
    This function is a subroutine of calc_nkvecs calculating k vectors for Ewald summation using combinatorial method.

    Arguments:
        sys_ (System): A System object representing the system to calculate k vectors for.
        nkmaxs (Array): A tuple containing the maximum number of k-vectors in each direction.

    Returns:
        nkvecs (Array): A numpy array containing the k vectors for Ewald summation. The shape of the array is (num_kvecs,3).
    """
    idxs0 = jnp.arange(-nkmaxs[0],+nkmaxs[0]+1)
    idxs1 = jnp.arange(-nkmaxs[1],+nkmaxs[1]+1)
    idxs2 = jnp.arange(-nkmaxs[2],+nkmaxs[2]+1)
    def vec(x,y,z): return jnp.array([x,y,z]) 
    vec1 = vmap(vec,  (0,None,None),0)
    vec2 = vmap(vec1, (None,0,None),0)
    vec3 = vmap(vec2, (None,None,0),0)
    nkvecs_ = vec3(idxs0,idxs1,idxs2)
    nkvecs  = jnp.reshape(nkvecs_,(-1,3))
    nkvecs = nkvecs[jnp.linalg.norm(nkvecs,axis=1)>1e-10,:] # omit (0,0,0)
    return nkvecs



