from scipy import constants as const
from math import pi
from jax import vmap, value_and_grad, grad, hessian, lax, jvp, custom_jvp, jacfwd, jit,random
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, gmres
from jax.scipy.linalg import solve
from scipy.optimize import minimize
import numpy as np
import delff.energy as energy
import delff.rtp as rtp
import delff.evalfunc as ef
from delff.objects import *
import copy
import optuna

twopi = 2*np.pi

def isfloat(parameter):
    """Checks if the input string can be converted to a float.

    Arguments:
      parameter (str): The string to check.

    Returns:
      bool: True if the string can be converted to a float, False otherwise.
    """
    if not parameter.isdecimal():
        try:
            float(parameter)
            return True
        except ValueError:
            return False
    else:
        return False

def _print(logpath,*args):
    """Prints the provided arguments to a log file and the console.

    Arguments:
      logpath (str): The path to the log file.
      *args: Variable length argument list of items to print.

    Returns:
      None
    """
    with open(logpath,mode='a') as f:
        for arg in args:
            f.write(str_)
        f.write('\n')
    print(str_)

def printff(ff_: ForceField):
    """Prints the attributes of a given ForceField object.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.

    Returns:
      None
    """
    print("\n## Force Field")
    print("\nbondtypes")
    for v in ff_.bondtypes: 
        print('%8.3f %8.3f'%(v[0],v[1]))

    print("\nangletypes")
    for v in ff_.angletypes:  print('%8.3f %8.3f'%(v[0],v[1]))

    print("\ndihdralks")
    for vec in ff_.dihedralks:  
        for v in vec: print('%8.3f'%v,end="")
        print('')

    print("\nimpropertypes")
    for v in ff_.impropertypes: print('%8.3f %8.3f %8.3f'%(v[0],v[1],v[2]))

    print("\npairs")
    for v in ff_.pairs: print('%8.3f %8.3f'%(v[0],v[1]))

    print("\ncharges")
    for i,v in enumerate(ff_.charges):  
        print('%8.3f '%v,end='')
        if i%10==9: print('')
    print('\ndielectric_constant',ff_.dielectric_constant)
    print('\nvscale3',ff_.vscale3)
    print('\ncscale3',ff_.cscale3)
    print('')


def print_jnpval(val):
    """Prints the jax.numpy object and numpy object adaptively

    Arguments:
      val: jax.numpy or numpy array/list/tuple object

    Returns:
      None
    """

    try:
        print('distvec_opt',val.primal)
    except:
        print('distvec_opt',val)

def doreg(ff_: ForceField, reg):
    """Regulates the valuables of a given ForceField object.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      reg (array-like): Regulation factors for each attribute.

    Returns:
      ForceField: A new ForceField object with regulated attributes.
    """
    bondtypes = ff_.bondtypes/reg[0:2]
    angletypes = ff_.angletypes/reg[2:4]
    dihedralks = ff_.dihedralks/jnp.tile(reg[5],(4))
    impropertypes = ff_.impropertypes/reg[6:9]
    pairs = ff_.pairs/reg[9:11]
    charges = ff_.charges/reg[11]
    return ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,
            ff_.dielectric_constant,ff_.vscale3,ff_.cscale3)


def dounreg(ff_: ForceField, reg):
    """Reverse Regularization of the valuables of a given ForceField object.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      reg (array-like): Regulation factors for each attribute.

    Returns:
      ForceField: A new ForceField object with regulated attributes.
    """
    return doreg(ff_,1/reg)


@jit
def logdamp(move: Array, scale: f64=1.) -> Array:
    """Applies a logarithmic damping to the input jax.numpy array.

    Arguments:
      move (Array): The input array.
      scale (float, optional): The scaling factor. Defaults to 1.

    Returns:
      Array: The damped array.
    """
    move /= scale
    damped = jnp.where(
        jnp.abs(move) > 1,
        jnp.log(1 + jnp.abs(move) * 1.72) * jnp.sign(move), move)
    move *= scale
    return damped

@jit
def logdamp_np(move: Array, scale: f64=1.) -> Array:
    """Applies a logarithmic damping to the input numpy array.

    Arguments:
      move (Array): The input array.
      scale (float, optional): The scaling factor. Defaults to 1.

    Returns:
      Array: The damped array.
    """
    move /= scale
    damped = np.where(
        np.abs(move) > 1,
        np.log(1 + np.abs(move) * 1.72) * np.sign(move), move)
    move *= scale
    return damped


def jnp2np(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,mask):
    """Converts multiple jax.numpy arrays into a single numpy array.

    Arguments:
      bondtypes, angletypes, dihedralks, impropertypes, pairs, charges (Array): 
        Jax numpy arrays representing various aspects of a molecular system.
      mask (Array): Boolean mask for array selection.

    Returns:
      ndarray: A single numpy array concatenating all input arrays.
    """
    # vervec must be numpy and raveled
    # nr means numpy and raveled
    bondtypes = np.asarray(bondtypes).ravel()
    angletypes = np.asarray(angletypes).ravel()
    dihedralks = np.asarray(dihedralks).ravel()
    impropertypes = np.asarray(impropertypes).ravel()
    pairs = np.asarray(pairs).ravel()
    charges = np.asarray(charges).ravel()
    vars_np = np.concatenate((bondtypes,angletypes,dihedralks,impropertypes,pairs,charges))
    vasr_np = mask*vars_np ### normalization
    vars_np = vars_np[mask>0]
    #vars_np = vars_np[mask==]
    return vars_np # returns np 1dim array

def convert_tsr4_to_mat2(tsr4,dim0=2,dim1=2):
    n0, n1, n2, n3 = 1, 1, 1, 1
    if dim0==2 and dim1==2:  n0,n1,n2,n3 = tsr4.shape
    elif dim0==1 and dim1==2:  
        n0,n2,n3 = tsr4.shape
    elif dim0==2 and dim1==1: 
        n0,n1,n2 = tsr4.shape
    elif dim0==1 and dim1==1:  
        n0,n2 = tsr4.shape
    else: print(tsr4.shape)
    return jnp.reshape(tsr4,(n0*n1,n2*n3))#n0*n1,n2*n3

def np2jnp(vars_new,nbondtypes,nangletypes,ndihedraltypes,nimpropertypes,npairs,natoms,vars_ini_full,mask):
    """Converts a jax array into multiple numpy arrays.

    Arguments:
      vars_new(Array): A single numpy array containing all masked parameters in a Force Field.
      nbondtypes,nangletypes,ndihedraltypes,nimpropertypes,npairs: numbers of the Force Field aprameters.
      natom: numbers of atoms in a molecule corresponding to the Force Field.
      mask (Array): Boolean mask for array selection.

    Returns:
      bondtypes, angletypes, dihedralks, impropertypes, pairs, charges (Array): 
        jax.numpy arrays representing various aspects of a molecular system.
    """
    # vervec must be numpy and raveled
    # nr means numpy and raveled

    vars_np = vars_ini_full
    vars_np[mask>0] = vars_new
 
    # bonds
    bgn,end = 0,nbondtypes*2
    bondtypes_np = vars_np[bgn:end]
    bondtypes = jnp.reshape(jnp.asarray(bondtypes_np),(nbondtypes,2))

    # angle
    bgn,end = end,end+nangletypes*2
    angletypes_np = vars_np[bgn:end]
    angletypes = jnp.reshape(jnp.asarray(angletypes_np),(nangletypes,2))

    # dihedral
    bgn,end = end,end+ndihedraltypes*4
    dihedralks_np = vars_np[bgn:end]
    dihedralks = jnp.reshape(jnp.asarray(dihedralks_np),(ndihedraltypes,4))

    # improper
    bgn,end = end,end+nimpropertypes*3
    impropertypes_np = vars_np[bgn:end]
    impropertypes = jnp.reshape(jnp.asarray(impropertypes_np),(nimpropertypes,3))

    # pairs
    bgn,end = end,end+npairs*2
    pairs_np = vars_np[bgn:end]
    pairs = jnp.reshape(jnp.asarray(pairs_np),(npairs,2))

    # charges
    bgn,end = end,end+natoms*2
    charges_np = vars_np[bgn:end]
    charges = jnp.reshape(jnp.asarray(charges_np),(natoms))

    return bondtypes, angletypes, dihedralks, impropertypes,  pairs, charges # returns jnp.arrays 


def make_lattice_replicas(sys_):
    """Generates lattice replicas of a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.

    Returns:
      Array: Coordinates of all atoms in the replicated system.
    """
    coord_0 = sys_.coord 
    nmol, natom, _ = coord_0.shape
    coord_1 = jnp.tile(coord_0,(27,1,1))
    tvmatunit = jnp.array([[ 0, 0, 0],[ 1, 0, 0],[-1, 0, 0],
                           [ 0, 1, 0],[ 1, 1, 0],[-1, 1, 0],
                           [ 0,-1, 0],[ 1,-1, 0],[-1,-1, 0],
                           [ 0, 0, 1],[ 1, 0, 1],[-1, 0, 1],
                           [ 0, 1, 1],[ 1, 1, 1],[-1, 1, 1],
                           [ 0,-1, 1],[ 1,-1, 1],[-1,-1, 1],
                           [ 0, 0,-1],[ 1, 0,-1],[-1, 0,-1],
                           [ 0, 1,-1],[ 1, 1,-1],[-1, 1,-1],
                           [ 0,-1,-1],[ 1,-1,-1],[-1,-1,-1]])
    tvmat = tvmatunit @ (sys_.lattice.transpose())
    tvtsr = jnp.reshape(tvmat,(27,1,1,3))
    tvtsr = jnp.tile(tvtsr,(1,nmol,natom,1))
    tvtsr = jnp.reshape(tvtsr,(27*nmol,natom,3))
    coord_1 = coord_1+tvtsr
    return coord_1

def get_reciprocal_lattice_and_V(lattice):
    """Calculates the reciprocal lattice and volume of a given lattice.
    ref: https://nanobuff.wordpress.com/2022/03/05/3d-and-2d-reciprocal-lattice-vectors-python-example/

    Arguments:
      lattice (Array): 3D array representing a lattice.

    Returns:
      tuple: A tuple containing the reciprocal lattice (Array) and volume (float).
    """
    a1 = lattice[0,:]
    a2 = lattice[1,:]
    a3 = lattice[2,:]
    b1 = twopi*jnp.cross(a2, a3)/jnp.dot(a1, jnp.cross(a2, a3))
    b2 = twopi*jnp.cross(a3, a1)/jnp.dot(a2, jnp.cross(a3, a1))
    b3 = twopi*jnp.cross(a1, a2)/jnp.dot(a3, jnp.cross(a1, a2))
    V  = jnp.dot(a1, jnp.cross(a2, a3))

    rec_lattice = jnp.zeros((3,3),f64)
    rec_lattice = rec_lattice.at[0,:].set(b1)
    rec_lattice = rec_lattice.at[1,:].set(b2)
    rec_lattice = rec_lattice.at[2,:].set(b3)
    return rec_lattice, V

