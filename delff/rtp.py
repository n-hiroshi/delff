from scipy import constants as const
from math import pi
from jax import vmap,jit
import jax.numpy as jnp
import numpy as np
from delff.objects import *
import jax.nn as jnn
from jax.experimental.host_callback import id_print

# Convert xyz sys_ to r(=bonds), theta(=angles), and phi(=dihedral and improper) sys_ination
@jit
def xyz2rtp(sys_: System, ffa_: ForceFieldAssignments ) -> RTPCoord:
    """
    Converts Cartesian coordinates to internal coordinates (bond lengths, angles, dihedrals and impropers) for a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      RTPCoord: RTPCoord object containing the internal coordinates.
    """
    rs     = bond_coord(sys_, ffa_)
    thetas = angle_coord(sys_, ffa_)
    phids  = dihedral_coord(sys_, ffa_)
    phiis  = improper_coord(sys_, ffa_)
    rall   = alldists(sys_,sys_,ffa_)
    return RTPCoord(rs,thetas,phids,phiis,rall)
     
@jit
def xyz2rtp_lattice(sys_: System, ffa_: ForceFieldAssignments ) -> RTPCoord:
    """
    Converts Cartesian coordinates to internal coordinates (all distances) in a periodic system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      RTPCoord: RTPCoord object containing the internal coordinates.
    """
    rs     = None
    thetas = None
    phids  = None
    phiis  = None

    rall   = alldists_neighbors(sys_,ffa_)
    return RTPCoord(rs,thetas,phids,phiis,rall)

@jit
def bond_coord(sys_: System, ffa_: ForceFieldAssignments ) -> Array:
    """
    Calculates the bond lengths for a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Bond lengths for each bond in the system.
    """
    def one_bond_coord(coord_,bond) -> f64:
        ibondtype, atom0, atom1 = bond[0], bond[1], bond[2]
        r = jnp.linalg.norm(coord_[atom0,:]-coord_[atom1,:])
        return r
    coord_ = sys_.coord
    return vmap(vmap(one_bond_coord,(None,0),0),(0,None),0)(coord_,ffa_.bonds)

@jit
def angle_coord(sys_: System,  ffa_: ForceFieldAssignments ) -> Array:
    """
    Calculates the bond angles for a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Bond angles for each angle in the system.
    """
    def one_angle_coord(coord_,angle) -> f64:
        iangletype, atom0, atom1, atom2 = angle[0], angle[1], angle[2], angle[3]
        rvec10 = coord_[atom1,:]-coord_[atom0,:]
        rvec12 = coord_[atom1,:]-coord_[atom2,:]
        r10 = jnp.linalg.norm(rvec10)
        r12 = jnp.linalg.norm(rvec12)
        theta = jnp.arccos(jnp.dot(rvec10,rvec12)/r10/r12)
        return theta
    coord_ = sys_.coord
    return vmap(vmap(one_angle_coord,(None,0),0),(0,None),0)(coord_,ffa_.angles)

@jit
def dihedral_coord(sys_: System, ffa_: ForceFieldAssignments ) -> Array:
    """
    Calculates the dihedral angles for a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Dihedral angles for each dihedral in the system.
    """
    def one_dihedral_coord(coord_,dihedral) -> f64:
        idihedraltype, atom0, atom1, atom2, atom3 = \
                dihedral[0], dihedral[1], dihedral[2], dihedral[3], dihedral[4]

        rvec10 = coord_[atom1,:]-coord_[atom0,:]
        rvec12 = coord_[atom1,:]-coord_[atom2,:]
        rvec23 = coord_[atom2,:]-coord_[atom3,:]

        crossA = jnp.cross(rvec10,rvec12)
        crossB = jnp.cross(-rvec12,rvec23)

        normA = jnp.linalg.norm(crossA)
        normB = jnp.linalg.norm(crossB)

        cos_phi = jnp.dot(crossA,crossB)/normA/normB

        vn1=1-1e-10
        cos_phi = jnp.where(cos_phi>-vn1,cos_phi,-vn1)
        cos_phi = jnp.where(cos_phi<+vn1,cos_phi,+vn1)
        phid = jnp.arccos(cos_phi)

        return phid

    coord_ = sys_.coord
    return vmap(vmap(one_dihedral_coord,(None,0),0),(0,None),0)(coord_,ffa_.dihedrals)

@jit
def improper_coord(sys_: System,  ffa_: ForceFieldAssignments ) -> Array:
    """
    Calculates the improper dihedral angles for a system.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Improper dihedral angles for each improper in the system.
    """
    def one_improper_coord(coord_,improper) -> f64:
        iimpropertype, atom0, atom1, atom2, atom3 = improper[0], improper[1], improper[2], improper[3], improper[4]

        rvec10 = coord_[atom1,:]-coord_[atom0,:]
        rvec12 = coord_[atom1,:]-coord_[atom2,:]
        rvec23 = coord_[atom2,:]-coord_[atom3,:]

        crossA = jnp.cross(rvec10,rvec12)
        crossB = jnp.cross(-rvec12,rvec23)

        normA = jnp.linalg.norm(crossA)
        normB = jnp.linalg.norm(crossB)
        cos_phi = jnp.dot(crossA,crossB)/normA/normB

        vn1=1-1e-10
        cos_phi = jnp.where(cos_phi>-vn1,cos_phi,-vn1)
        cos_phi = jnp.where(cos_phi<+vn1,cos_phi,+vn1)
        phii = jnp.arccos(cos_phi)
 
        return phii

    coord_ = sys_.coord
    return vmap(vmap(one_improper_coord,(None,0),0),(0,None),0)(coord_,ffa_.impropers)


@jit
def alldists(sys_0: System, sys_1: System, ffa_: ForceFieldAssignments) -> Array:
    """
    Calculates the distances between all atom pairs in a system.

    Arguments:
      sys_0 (System): System object containing the atomic coordinates.
      sys_1 (System): Another System object containing atomic coordinates. Usually, sys_0 is identical to sys_1.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Distances between all atom pairs in the system.
    """

    def dist(v0,v1):
        dv = v0-v1
        return jnp.sqrt(jnn.relu(jnp.dot(dv,dv)))

    vdist1 = vmap(dist  ,(None,0),0)
    vdist2 = vmap(vdist1,(None,0),0)
    vdist3 = vmap(vdist2,(0,None),0)
    vdist4 = vmap(vdist3,(0,None),0)
    rall   = vdist4(sys_0.coord,sys_1.coord)
    return rall


@jit
def alldists_neighbors(sys_: System, ffa_: ForceFieldAssignments) -> Array:
    """
    Calculates the distances between neighboring atoms in a system, considering periodic boundary conditions.

    Arguments:
      sys_ (System): System object containing the atomic coordinates.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom.

    Returns:
      Array: Distances between neighboring atom pairs in the system.
    """

    def dist(neighbors_each,sys_):
        imol,iatom,jcellx,jcelly,jcellz,jmol,jatom = neighbors_each
        vi = sys_.coord[imol,iatom,:]
        vj = sys_.coord[jmol,jatom,:]
        tv = jnp.asarray([jcellx,jcelly,jcellz]) @ sys_.lattice
        dv = vi - (vj+tv)
        return jnp.sqrt(jnn.relu(jnp.dot(dv,dv)))

    neighbors = ffa_.neighbors
    vdist = vmap(dist  ,(0,None),0)
    rall  = vdist(neighbors,sys_)

    return rall






