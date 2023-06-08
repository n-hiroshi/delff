from scipy import constants as const
from math import pi
from jax import vmap, value_and_grad, grad, hessian, lax, jvp, custom_jvp, jacfwd, jit, jacrev
#import jaxopt.NonlinearCG as NCG
import jaxopt 
import jax.nn as jnn
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, gmres
from jax.scipy.linalg import solve
from scipy.optimize import minimize
import numpy as np
import delff.lammpsff as lff
import delff.gaussianff as gff
from delff.objects import *
import delff.energy as energy
from jax.experimental.host_callback import id_print

@custom_jvp
def opt_lat_sys(ff_,sys_int,ffa_):
    '''
    This function is a wrapper around another function opt_lat_sys_jaxopt. 
    It takes in three arguments and returns the result of opt_lat_sys_jaxopt function.

    Args:

    - ff_: A set of force field parameters for the system.
    - sys_int: A System object representing the initial configuration of the system.
    - ffa_: A set of additional parameters.

    Returns:

    - sys_opt: A System object representing the optimized configuration of the system.
    - Note: The @custom_jvp decorator indicates that this function 
      has a custom JVP rule defined for it.
    '''
    sys_opt = opt_lat_sys_jaxopt(ff_,sys_int,ffa_)
    return sys_opt

@jit
@opt_lat_sys.defjvp
def opt_lat_sys_jvp(primals, tangents):
    '''
    This function is the JVP rule for the optimization 
    of the lattice system of a molecular simulation.
    It calculates the Jacobian-vector product (JVP)
    of the function opt_lat_sys_jaxopt using implicit differentiation.
    
    Args:
        - primals: a tuple of three inputs: ff_ (force field parameters), 
        sys_int (an instance of the System class that contains the initial configuration
        of the molecular system), and ffa_ (force field attributes).

        - tangents: a tuple of three tangents that correspond to the derivatives 
        of ff_, sys_int, and ffa_, respectively.

    Returns:
        - primals_out: an instance of the System class that contains the optimized
        configuration of the molecular system.

        - tangents_out: a tuple of three tangents that correspond to the derivatives 
        of ff_, sys_int, and ffa_ with respect to the optimized configuration 
        of the molecular system.
    '''

    def sys2coordlat(ff_,sys_,ffa_):
        coordlat = jnp.reshape(sys_.coord,(-1,3))
        coordlat = jnp.concatenate((coordlat,sys_.lattice))
        return coordlat

    def coordlat2sys(ff_,coordlat,ffa_,sys_int):
        nmol,natom,_ = sys_int.coord.shape
        sys_=System(jnp.reshape(coordlat[:-3,:],(nmol,natom,3)), \
                    jnp.reshape(coordlat[-3:,:],(3,3)))
        return sys_

    def __energy(ff_,coordlat,ffa_,sys_int):
        sys_ = coordlat2sys(ff_,coordlat,ffa_,sys_int)
        return energy.energy_coord(ff_,sys_,ffa_)

    ff_, sys_int, ffa_ = primals
    dff_, dsys_int, dffa_ = tangents

    sys_opt = opt_lat_sys_jaxopt(ff_,sys_int,ffa_)
    coordlat  = sys2coordlat(ff_, sys_opt,ffa_)
    #dcoordlat = sys2coordlat(ff_,dsys_int,ffa_)

    PDE = grad(__energy,argnums=1)
    dF_dcoordlat = jacrev(PDE,argnums=1)(ff_, coordlat, ffa_, sys_int)
    dF_dff_     = jacrev(PDE,argnums=0)(ff_, coordlat, ffa_, sys_int)
 
    #nmol,natoms,_,_,_,_ = dF_dsys_opt.coord.coord.shape
    ntot = coordlat.shape[0]
    dim2_dF_dcoordlat      = jnp.reshape(dF_dcoordlat,(ntot*3,ntot*3))
    dim2_dF_dbondtypes     = jnp.reshape(dF_dff_.bondtypes,(ntot*3,-1))
    dim2_dF_dangletypes    = jnp.reshape(dF_dff_.angletypes,(ntot*3,-1))
    dim2_dF_ddihedralks    = jnp.reshape(dF_dff_.dihedralks,(ntot*3,-1))
    dim2_dF_dimpropertypes = jnp.reshape(dF_dff_.impropertypes,(ntot*3,-1))
    dim2_dF_dpairs         = jnp.reshape(dF_dff_.pairs,(ntot*3,-1))
    dim2_dF_dcharges       = jnp.reshape(dF_dff_.charges,(ntot*3,-1))
 
    dim1_dbondtypes = dff_.bondtypes.ravel()
    dim1_dangletypes = dff_.angletypes.ravel()
    dim1_ddihedralks = dff_.dihedralks.ravel()
    dim1_dimpropertypes = dff_.impropertypes.ravel()
    dim1_dpairs = dff_.pairs.ravel()
    dim1_dcharges = dff_.charges.ravel()
 
    # Implicit Function Differentiation
    # d2E/dudu @ X = -d2E/dpdu 
    b  = -dim2_dF_dbondtypes @ dim1_dbondtypes
    b += -dim2_dF_dangletypes @ dim1_dangletypes
    b += -dim2_dF_ddihedralks @ dim1_ddihedralks
    b += -dim2_dF_dimpropertypes @ dim1_dimpropertypes
    b += -dim2_dF_dpairs @ dim1_dpairs
    b += -dim2_dF_dcharges @ dim1_dcharges
    A  =  dim2_dF_dcoordlat # d2E/dudu

    L, U = ilu(A)
    z = solve(L,b,lower=True)
    x = solve(U,z,lower=False)

    dim1_dcoordlat = x#cg(A,b,tol=1e-7)[0]
    dim2_dcoordlat = jnp.reshape(dim1_dcoordlat,(-1,3))
    dsys_opt       = coordlat2sys(ff_,dim2_dcoordlat,ffa_,sys_int)
 
    primals_out  = sys_opt
    tangents_out = dsys_opt
    return primals_out, tangents_out

@jit
def opt_lat_sys_jaxopt(ff_,sys_int,ffa_):
    '''
    This function is used for optimizing the energy of a system 
    using JAXOpt library for gradient descent optimization.
    
    Args:
        - ff_: a function that describes the force field of the physical system.
        - sys_int: an initial state of the physical system, 
          which contains information about the positions and momenta of its particles.
        - ffa_: a set of additional parameters needed
          for calculating the energy of the physical system.

    Returns:
        - sys_opt: the optimized state of the physical system, 
          which minimizes its energy.
        - info: information about the optimization process,
          such as the number of iterations and the final value of the energy.
    '''
    
    def func_(sys_int,ff_,ffa_): 
        energy_ =  energy.energy_coord(ff_,sys_int,ffa_)
        return energy_
    SPM_ = jaxopt.GradientDescent(fun=func_,tol=0.001, jit=True)
    sys_opt,info = SPM_.run(sys_int,ff_,ffa_)
    return sys_opt

        
@jit
def ilu(mat: Array) -> Array:
    '''
    This function performs an incomplete LU decomposition on a given matrix, 
    which factorizes the matrix into a lower triangular matrix
    and an upper triangular matrix with the same diagonal as the original matrix.

    Args:
        - mat: a 2D array representing the matrix to be decomposed.

    Returns:
        - L: a lower triangular matrix with the same diagonal as mat.
        - U: an upper triangular matrix with the same diagonal as mat.
    '''

    n = mat.shape[0]

    def kloop(mat, k):
        def iloop(mat,i):

            def processrow(i):
                mat = mat.at[i,k].set(mat[i,k]/mat[k,k])
                def jloop(mat,j):
                    mat=lax.cond(jnp.logical_and(mat[i,j]!=0,j>=k+1),
                                 lambda j: mat.at[i,j].set(mat[i,j]-mat[i,k]*mat[k,j]),
                                 lambda j: mat,j)
                    return mat,None
                mat,_ = lax.scan(iloop,mat,jnp.arange(n))

            return mat,None

            mat=lax.cond(jnp.logical_and(mat[i,k]!=0,i>=k+1),
                         lambda i: processrow(i),
                         lambda i: mat,i)

        mat,_ = lax.scan(iloop,mat,jnp.arange(n))
        return mat,None
    mat,_ = lax.scan(kloop,mat,jnp.arange(n-1))



    # Lower trianglular and unit diagonal
    Lcol = vmap(lambda i,j,v: jnp.where(i>j,v,0),(0,None,0),0)
    L = vmap(Lcol,(None,0,1),1)(jnp.arange(n),jnp.arange(n),mat)
    L+=jnp.eye(n)

    # Upper trianglular and the same diagonals as mat
    U = jnp.zeros_like(mat)
    Ucol = vmap(lambda i,j,v: jnp.where(i<=j,v,0),(0,None,0),0)
    U = vmap(Ucol,(None,0,1),1)(jnp.arange(n),jnp.arange(n),mat)

    return L, U

