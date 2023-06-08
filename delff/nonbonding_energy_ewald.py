from scipy import constants as const
#from math import pi
from jax import vmap,jit,grad,jacfwd,lax
import jax.numpy as jnp
from jax.scipy.special import erfc,erf
import numpy as np
from delff.objects import *
from delff.util import * #get_reciprocal_lattice_and_V
#from delff.rtp import xyz2rtp,xyz2rtp_lattice
from delff.nonbonding_energy import nonbond_energy_intramol_correction,charmm_Sfactor,calc_vdw,calc_coulomb

rt2 = jnp.sqrt(2)
pi = np.pi

@jit
def nonbond_energy_ewald(ff_ : ForceField,
                         rtp_: RTPCoord,
                         ffa_: ForceFieldAssignments,                            
                         rtp_lat: RTPCoord,
                         sys_:System) -> f64:
    """Calculates the non-bonded energy of a system using Ewald summation.

    The function first computes the non-bonded energy without intra-molecular correction, 
    then it calculates the intra-molecular correction and subtracts it from the total non-bonded energy.

    Arguments:
        ff_ (ForceField): ForceField object containing the parameters for the force field.
        rtp_ (RTPCoord): RTPCoord object containing the system coordinates in rtp format.
        ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
        rtp_lat (RTPCoord): RTPCoord object containing the lattice coordinates in rtp format.
        sys_ (System): System object containing the information for the system.

    Returns:
        float: The non-bonded energy of the system using Ewald summation in Ha.
    """

    E_coul, E_long, E_vdw  = nonbond_energy_ewald_wo_intramol_correction(ff_,rtp_lat,ffa_,sys_) 
    cets,vets = nonbond_energy_intramol_correction(ff_,rtp_,ffa_)
    E_coul -= cets
    E_vdw  -= vets
    return E_coul, E_long, E_vdw

@jit
def calc_coulomb_short(r,atomtype0,atomtype1,charges,dielectric_constant,ccoeff,ccutoff,sigma):
    """
    Calculates the short-range Coulomb energy between two atoms.

    Arguments:
      r (float): Distance between two atoms.
      atomtype0 (int): Atom type of the first atom.
      atomtype1 (int): Atom type of the second atom.
      charges (array): Array containing the charges of each atom type.
      dielectric_constant (float): The constant representing the dielectric properties of the medium.
      ccoeff (float): The coefficient for the short-range part of the interaction.
      ccutoff (float): The cutoff distance for the short-range part of the interaction.
      sigma (float): The Ewald summation parameter.

    Returns:
      float: Short-range Coulomb energy.
    """
    charge0 = charges[atomtype0]
    charge1 = charges[atomtype1]
    r = jnp.where(r>1e-10,r,1e-10)
    ccoeff = f64(r<ccutoff)*f64(r>1e-3)*ccoeff
    return jnp.where(ccoeff>1e-3, \
            ccoeff*charge0*charge1/dielectric_constant/r*erfc(r/rt2/sigma),0) 

@jit 
def calc_coulomb_long_all(ff_,sys_,ffa_):
    """
    Calculates the long-range Coulomb energy of the system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_ (System): System object containing the lattice and coordinates of the system.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Long-range Coulomb energy of the system.
    """
    
    sigma=ffa_.sigma_ewald
    rec_lattice ,V = get_reciprocal_lattice_and_V(sys_.lattice)
    kvecs = ffa_.nkvecs @ rec_lattice

    def q_exp_ikr(charge,rvec,kvec):
        kr = jnp.dot(kvec,rvec)
        q_exp_ikr = charge * jnp.exp(1j*kr)
        return q_exp_ikr
    
    def structure_factor_2norm(charges,coord,kvec):
        v1 = vmap(q_exp_ikr,(   0,0,None),0) # atoms in a mol
        v2 = vmap(v1       ,(None,0,None),0) # mols in a unit cell
        v2val = v2(charges,coord,kvec)
        v2sum =  jnp.sum(jnp.sum(v2val))
        return jnp.real(v2sum*jnp.conj(v2sum))

    def ksumed_func(charges,coord,kvec,sigma):
        k2 = jnp.linalg.norm(kvec)**2
        prefactor = jnp.exp( - sigma**2 * k2 /2 ) / k2
        return prefactor * structure_factor_2norm(charges,coord,kvec)

    charges = ff_.charges[ffa_.atomtypes]

    coulomb_energy_long_ = jnp.sum(                \
        vmap(ksumed_func,(None,None,0,None),0)     \
        (charges,sys_.coord,kvecs,sigma))

    relative_prefactor_long2short = 2*pi/V/ff_.dielectric_constant

    return relative_prefactor_long2short*coulomb_energy_long_

@jit 
def calc_coulomb_self(ff_,sys_,ffa_):
    """
    Calculates the self-interaction part of the Coulomb energy.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_ (System): System object containing the lattice and coordinates of the system.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      float: Self-interaction part of the Coulomb energy.
    """
    sigma = ffa_.sigma_ewald
    nmol = sys_.coord.shape[0]
    sqsum_charge = jnp.sum(ff_.charges[ffa_.atomtypes]**2) * nmol
    return 1/jnp.sqrt(2.0*pi)/sigma*sqsum_charge/ff_.dielectric_constant


@jit
def nonbond_energy_ewald_wo_intramol_correction(
                         ff_ : ForceField,
                         rtp_: RTPCoord, # must be neghbors list type 
                         ffa_: ForceFieldAssignments, # must include a neighbors-property 
                         sys_: System
                         ) -> f64:
    """
    Calculates the non-bonded energy of the system using Ewald summation, without intramolecular correction.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in RTP format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      sys_ (System): System object containing the lattice and coordinates of the system.

    Returns:
      float: Non-bonded energy of the system.
    """
    sigma=ffa_.sigma_ewald
    # short range
    def calc_coulomb_neighbors(r,neighbors_each,atomtypes,\
            charges,dielectric_constant,ccutoff,sigma):
        ccoeff = jnp.where(jnp.sum(jnp.abs(neighbors_each[2:5]))==0,1,0.5)
        return calc_coulomb_short(r,atomtypes[neighbors_each[1]],atomtypes[neighbors_each[6]],\
                            charges,dielectric_constant,ccoeff,ccutoff,sigma)

    v1cnei = vmap(calc_coulomb_neighbors,(   0,   0,None,None,None,None,None),0) 
    coulomb_energy_allpairs = v1cnei(rtp_.rall,ffa_.neighbors,ffa_.atomtypes,
                                     ff_.charges,ff_.dielectric_constant,ffa_.ccutoff,sigma)
    coulomb_energy_short = jnp.sum(jnp.sum(jnp.sum(jnp.sum(coulomb_energy_allpairs))))

    # long range
    coulomb_energy_long  = calc_coulomb_long_all(ff_,sys_,ffa_)

    # self energy
    coulomb_energy_self  = calc_coulomb_self(ff_,sys_,ffa_)

    # sum up
    E_coul = coulomb_energy_short
    E_long = coulomb_energy_long - coulomb_energy_self

    def calc_vdw_neighbors(r,neighbors_each,atomtypes,pairs,vcutoff):
        vcoeff = jnp.where(jnp.sum(jnp.abs(neighbors_each[2:5]))==0,1,0.5)
        return calc_vdw(r,atomtypes[neighbors_each[1]],atomtypes[neighbors_each[6]],
                        pairs,vcoeff,vcutoff)

    v1vnei = vmap(calc_vdw_neighbors,(   0,   0,None,None,None),0) 
    vdw_energy_allpairs = v1vnei(rtp_.rall,ffa_.neighbors,ffa_.atomtypes,ff_.pairs,ffa_.vcutoff)
    E_vdw = jnp.sum(jnp.sum(jnp.sum(jnp.sum(vdw_energy_allpairs))))

    return E_coul, E_long, E_vdw 

