from scipy import constants as const
from math import pi
from jax import vmap,jit,grad,jacfwd,lax
import jax.numpy as jnp
import numpy as np
from delff.objects import *
from delff.rtp import xyz2rtp,xyz2rtp_lattice
from jax.experimental.host_callback import id_print

@jit
def bond_energy(ff_ : ForceField,
                rtp_: RTPCoord,
                ffa_: ForceFieldAssignments ) -> f64:

    """Calculates the total bond energy of a system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Total bond energy of the system in Ha.
    """

    def one_bond_energy(r,ibondtype,bondtypes) -> f64:
        k, req = bondtypes[ibondtype,0],bondtypes[ibondtype,1]
        return k*(r-req)**2

    mol_bond_energy = vmap(one_bond_energy,(0,   0,None),0) # atom mapping  
    all_bond_energy = vmap(mol_bond_energy,(0,None,None),0) # mol mapping
    tot_bond_energy = jnp.sum(jnp.sum(all_bond_energy(rtp_.rs,ffa_.bonds[:,0],ff_.bondtypes)))
    return tot_bond_energy

@jit
def angle_energy(ff_ : ForceField,
                 rtp_: RTPCoord,
                 ffa_: ForceFieldAssignments ) -> f64:
    """Calculates the total angle energy of a system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Total angle energy of the system in Ha.
    """

    def one_angle_energy(theta,iangletype,angletypes) -> f64:
        k, thetaeq = angletypes[iangletype,0],angletypes[iangletype,1]
        return k*(theta-thetaeq*(np.pi/180.0))**2

    mol_angle_energy = vmap(one_angle_energy,(0,   0,None),0)
    all_angle_energy = vmap(mol_angle_energy,(0,None,None),0)
    tot_angle_energy = jnp.sum(jnp.sum(   \
       all_angle_energy(rtp_.thetas,ffa_.angles[:,0],ff_.angletypes)))
    return tot_angle_energy

@jit
def dihedral_energy(ff_ : ForceField,
                    rtp_: RTPCoord,
                    ffa_: ForceFieldAssignments ) -> f64:

    """Calculates the total dihedral energy of a system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Total dihedral energy of the system in Ha.
    """

    def one_dihedral_energy(phi,idihedraltype,dihedralmasks,dihedralks,dihedralphis) -> f64:
        vs = dihedralks[idihedraltype,:]
        phieqs = dihedralphis[idihedraltype,:]
        edihed  =  dihedralmasks[idihedraltype,0]*vs[0]*(1+jnp.cos(1*phi-phieqs[0]*(np.pi/180)))
        edihed  += dihedralmasks[idihedraltype,1]*vs[1]*(1+jnp.cos(2*phi-phieqs[1]*(np.pi/180)))
        edihed  += dihedralmasks[idihedraltype,2]*vs[2]*(1+jnp.cos(3*phi-phieqs[2]*(np.pi/180)))
        edihed  += dihedralmasks[idihedraltype,3]*vs[3]*(1+jnp.cos(4*phi-phieqs[3]*(np.pi/180)))
        return edihed

    mol_dihedral_energy = vmap(one_dihedral_energy,(0,   0,None,None,None),0)
    all_dihedral_energy = vmap(mol_dihedral_energy,(0,None,None,None,None),0)
    tot_dihedral_energy = jnp.sum(jnp.sum(all_dihedral_energy(rtp_.phids,ffa_.dihedrals[:,0],\
        ffa_.dihedralmasks,ff_.dihedralks,ffa_.dihedralphis)))
    return tot_dihedral_energy

@jit
def improper_energy(ff_ : ForceField,
                    rtp_: RTPCoord,
                    ffa_: ForceFieldAssignments ) -> f64:

    """Calculates the total improper energy of a system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Total improper energy of the system in Ha.
    """

    def one_improper_energy(phi,iimpropertype,impropertypes) -> f64:
        mag,po,period = impropertypes[iimpropertype,0],\
                        impropertypes[iimpropertype,1],\
                        impropertypes[iimpropertype,2]
        e_improper = mag*(1+jnp.cos(period*phi-po*(np.pi/180.0))) 
        return e_improper

    mol_improper_energy = vmap(one_improper_energy,(0   ,0,None),0)
    all_improper_energy = vmap(mol_improper_energy,(0,None,None),0)
    tot_improper_energy = jnp.sum(jnp.sum(     \
        all_improper_energy(rtp_.phiis,ffa_.impropers[:,0],ff_.impropertypes)))
    return tot_improper_energy


