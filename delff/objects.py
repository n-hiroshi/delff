#objects
from delff import dataclasses
from jax import numpy as jnp
from typing import Union

Array = jnp.array
f64 = jnp.float64
i32 = jnp.int32
i16 = jnp.int16
qqr2e = 0.529178
kcalM2Ha = 627.50960803


@dataclasses.dataclass
class ForceField: 
    """
    Defines the parameters for a molecular force field.

    Arguments:
      bondtypes (Array, optional): Bond parameters of the force field. Default is None.
      angletypes (Array, optional): Angle parameters of the force field. Default is None.
      dihedralks (Array, optional): Dihedral parameters of the force field. Default is None.
      impropertypes (Array, optional): Improper dihedral parameters of the force field. Default is None.
      pairs (Array, optional): Nonbonded interaction parameters of the force field. Default is None.
      charges (Array, optional): Atomic charges for the force field. Default is None.
      dielectric_constant (Array, optional): Dielectric constant for the force field. Default is 3.0.
      vscale3 (f64, optional): Scaling factor for 1-3 van der Waals interactions. Default is 1.0.
      cscale3 (f64, optional): Scaling factor for 1-3 Coulomb interactions. Default is 1.0.
    """
    bondtypes: Array=None 
    angletypes: Array=None 
    dihedralks: Array=None 
    impropertypes: Array=None  
    pairs: Array=None 
    charges: Array=None
    dielectric_constant: Array=3.0
    vscale3: f64=1.0 # special_bonds dreiding
    cscale3: f64=1.0 # special_bonds dreiding


@dataclasses.dataclass
class System:
    """
    Defines the system of interest, including atomic coordinates and lattice parameters.

    Arguments:
      coord (Array, optional): Atomic coordinates in the system. Default is None.
      lattice (Array, optional): Lattice parameters for periodic systems. Default is None.
    """
    coord: Array=None
    lattice: Array=None

@dataclasses.dataclass
class ForceFieldAssignments: 
    """
    Contains the assignment of force field parameters for each atom in the system.

    Arguments:
      atomtypes (Array, optional): Atom type indices for each atom in the system. Default is None.
      masses (Array, optional): Atomic masses for each atom in the system. Default is None.
      bonds (Array, optional): Bond parameters for each atom in the system. Default is None.
      angles (Array, optional): Angle parameters for each atom in the system. Default is None.
      dihedrals (Array, optional): Dihedral parameters for each atom in the system. Default is None.
      dihedralmasks (Array, optional): Masks for applying dihedral parameters. Default is None.
      dihedralphis (Array, optional): Phi angles for dihedrals. Default is None.
      impropers (Array, optional): Improper dihedral parameters for each atom in the system. Default is None.
      adjmat012 (Array, optional): Adjacency matrix for the topological distances of zero to two. Default is None. The zero distance indicates the same atom.
      adjmat3 (Array, optional): Adjacency matrix for the topological distance of three. Default is None.
      nmolvec (Array, optional): Vector containing the number of molecules in the system. Default is None.
      natomvec (Array, optional): Vector containing the number of atoms in each molecule. Default is None.
      intermol_dists (Array, optional): Inter-molecular distances. Default is None.
      nbatomtypesets (list, optional): List of non-bonded atom type sets. Default is None.
      neighbors (Array, optional): Matrix containing the neighbors for each atom. Default is None.
      latidx (Array, optional): Matrix containing the lattice indices for each atom. Default is None.
      nkvecs (Array, optional): Matrix containing the k vectors for Ewald summation. Default is None.
      nkmaxs (Array, optional): Maximum k values for each direction. Default is None.
      alpha_ewald (f64, optional): Alpha parameter for Ewald summation. Default is None.
      sigma_ewald (f64, optional): Sigma parameter for Ewald summation. Default is None.
      vcutoff (i32, optional): Cut-off radius for van der Waals interactions. Default is 10.
      ccutoff (i32, optional): Cut-off radius for Coulomb interactions. Default is 10.
    """
    atomtypes: Array=None 
    masses: Array=None 
    bonds: Array=None 
    angles: Array=None 
    dihedrals: Array=None 
    dihedralmasks: Array=None 
    dihedralphis: Array=None 
    impropers: Array=None 
    adjmat012: Array=None 
    adjmat3  : Array=None 
    nmolvec: Array=None 
    natomvec: Array=None
    intermol_dists: Array=None
    nbatomtypesets: list=None
    neighbors: Array=None # for neighbor list
    latidx: Array=None    # for neighbor list
    nkvecs: Array=None # for Ewald sum.
    nkmaxs: Array=None # for Ewald sum.
    alpha_ewald: f64=None # for Ewald sum. 
    sigma_ewald: f64=None # sigma_ewald = 1/alpha_ewald/sqrt(2) # Lee&Cai Ewald Sum paper
    vcutoff: i32=10 
    ccutoff: i32=10 #=rcut https://wanglab.hosted.uark.edu/DLPOLY2/node114.html

@dataclasses.dataclass
class Task: 
    """
    Defines a computational task to perform on a system.

    Arguments:
      Ltype (str, optional): Type of task to perform. Default is 'structures'.
      params (dict, optional): Parameters required for the task. Default is None.
    """
    Ltype: str='structures'
    params: dict=None

@dataclasses.dataclass
class RTPCoord: 
    """
    Defines the internal coordinates of a molecule in terms of bond lengths, angles, and torsions.

    Arguments:
      rs (Array, optional): Bond lengths for each bond in the system. Default is None.
      thetas (Array, optional): Bond angles for each angle in the system. Default is None.
      phids (Array, optional): Dihedral angles for each dihedral in the system. Default is None.
      phiis (Array, optional): Improper dihedral angles for each improper in the system. Default is None.
      rall (Array, optional): Distances between all atom pairs in the system. Default is None.
    """
    rs: Array=None # bond length
    thetas: Array=None # angle
    phids: Array=None # dihedral
    phiis: Array=None # improper dihedral
    rall: Array=None # all of dists of all atom pairs

def update(obj: Union[ForceField, ForceFieldAssignments], **kwargs) -> Union[ForceField, ForceFieldAssignments]:

    return obj.__class__(
        **{
            key: kwargs[key] if key in kwargs else value
            for key, value in obj.__dict__.items()
        })
              
