from scipy import constants as const
from math import pi
from jax import vmap,jit,grad,jacfwd,lax
import jax.numpy as jnp
import numpy as np
from delff.objects import *
from delff.rtp import xyz2rtp,xyz2rtp_lattice
from delff.bonding_energy import bond_energy, angle_energy, dihedral_energy, improper_energy
from delff.nonbonding_energy import nonbond_energy
from delff.nonbonding_energy_neighbors import nonbond_energy_neighbors
from delff.nonbonding_energy_ewald import nonbond_energy_ewald
from jax.experimental.host_callback import id_print

#@jit
def energy(ff_: ForceField,
           rtp_: RTPCoord,
           ffa_: ForceFieldAssignments,
           ewald: bool=True) -> f64:
    """Calculates the total energy of a system given the force field, coordinates in rtp format, and force field assignments.
    
    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      ewald (bool, optional): If True, uses Ewald summation for long-range electrostatic interactions. Defaults to True.
 
    Returns:
      float: Total energy of the system in Ha.
    """

    E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw = \
           energy_each(ff_, rtp_, ffa_)

    return E_bond+E_angle+E_dihed+E_improper+E_coul+E_long+E_vdw 

#@jit
def energy_coord(ff_: ForceField,
           sys_: System,
           ffa_: ForceFieldAssignments,
           ewald: bool=True) -> f64:
    """Calculates the total energy of a system given the force field, coordinates in cartesian format, and force field assignments.

    Args:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_ (System): System object containing the coordinates in cartesian format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      ewald (bool, optional): If True, uses Ewald summation for long-range electrostatic interactions. Defaults to True.

    Returns:
      float: Total energy of the system in Ha.
    """


    E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw = \
           energy_each_coord(ff_, sys_, ffa_,ewald)
    return E_bond+E_angle+E_dihed+E_improper+E_coul+E_long+E_vdw 

#@jit
def energy_each_coord(ff_: ForceField,
           sys_: System,
           ffa_: ForceFieldAssignments,
           ewald: bool=True):
    """Calculates the energy contribution of each interaction type in a system given the force field, coordinates in cartesian format, and force field assignments.
    Define nonbonding type for a periodic tystem with ewald option
    ewald=False: lj/charmm/coul/charmm
    ewald=True : lj/charmm/coud/long with pair ewald option
    ref: https://docs.lammps.org/pair_charmm.html

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_ (System): System object containing the coordinates in cartesian format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      ewald (bool, optional): If True, uses Ewald summation for long-range electrostatic interactions. Defaults to True.

    Returns:
      Tuple[float]: Tuple containing the energy contribution of each interaction type (bonding, angle, dihedral, improper, Coulombic, long-range Coulombic, and van der Waals) in Ha.
    """


    rtp_ = xyz2rtp(sys_,ffa_)

    if sys_.lattice is None:
        E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw = \
           energy_each(ff_, rtp_, ffa_)
        E_long=0.0
    elif ewald:
        #print("Coulomb: Ewald")
        rtp_lat = xyz2rtp_lattice(sys_,ffa_) # neighbors list format
        # ewald methods requires 3-dim coord and lattice for a long-range energy
        E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw = \
           energy_each(ff_, rtp_, ffa_, rtp_lat, ewald, sys_)
    else:
        #print("Coulomb: Charmm")
        rtp_lat = xyz2rtp_lattice(sys_,ffa_) # neighbors list format
        E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw = \
           energy_each(ff_, rtp_, ffa_, rtp_lat, ewald)
        E_long=0.0

    return E_bond,E_angle,E_dihed,E_improper,E_coul,E_long,E_vdw 

#@jit
def energy_each(ff_ : ForceField,
                rtp_: RTPCoord,
                ffa_: ForceFieldAssignments,
                rtp_lat: RTPCoord=None,
                ewald: bool=True,
                sys_: System=None) -> f64:

    """Calculates the energy contribution of each interaction type in a system given the force field, coordinates in rtp format, and force field assignments.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      rtp_lat (RTPCoord, optional): RTPCoord object containing the coordinates in rtp format for the lattice sites of a periodic system. Defaults to None.
      ewald (bool, optional): If True, uses Ewald summation for long-range electrostatic interactions. Defaults to True.

    Returns:
      Tuple[float]: Tuple containing the energy contribution of each interaction type (bonding, angle, dihedral, improper, Coulombic, long-range Coulombic, and van der Waals) in Ha.
    """

    assert len(ffa_.nmolvec)==1 # FF for multi-moltypes is not implemented

    if len(ffa_.bonds)>=1: E_bond = bond_energy(ff_,rtp_,ffa_)/kcalM2Ha
    else: E_bond=0
    if len(ffa_.angles)>=1: E_angle = angle_energy(ff_,rtp_,ffa_)/kcalM2Ha
    else: E_angle=0

    if len(ffa_.dihedrals)>=1: E_dihedral = dihedral_energy(ff_,rtp_,ffa_)/kcalM2Ha
    else: E_dihedral = 0
    if len(ffa_.impropers)>=1: 
        E_improper = improper_energy(ff_,rtp_,ffa_)/kcalM2Ha
    else: E_improper = 0

    if rtp_lat is None:
        E_coul, E_vdw =  nonbond_energy(ff_,rtp_,ffa_)
        E_long = 0.0
    elif ewald:
        E_coul, E_long, E_vdw =  nonbond_energy_ewald(ff_,rtp_,ffa_,rtp_lat,sys_)
    else:
        E_coul, E_vdw =  nonbond_energy_neighbors(ff_,rtp_,ffa_,rtp_lat)
        E_long = 0.0

    E_vdw = E_vdw/kcalM2Ha
    E_coul =qqr2e*E_coul
    E_long =qqr2e*E_long
    return E_bond,E_angle,E_dihedral,E_improper,E_coul,E_long,E_vdw

     
