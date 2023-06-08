from scipy import constants as const
from math import pi
from jax import vmap,jit,grad,jacfwd,lax
import jax.numpy as jnp
import numpy as np
from delff.objects import *
from delff.rtp import xyz2rtp,xyz2rtp_lattice
from delff.nonbonding_energy import nonbond_energy_intramol_correction, charmm_Sfactor, calc_coulomb, calc_vdw


@jit
def nonbond_energy_neighbors(ff_ : ForceField,
                          rtp_: RTPCoord,
                          ffa_: ForceFieldAssignments,                            
                          rtp_lat: RTPCoord) -> f64:
    """
    Calculates the non-bonded energy between neighboring atoms, taking into account both external and internal interactions.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in RTP format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      rtp_lat (RTPCoord): RTPCoord object containing the lattice coordinates in RTP format.

    Returns:
      tuple: Tuple containing the Coulomb and van der Waals energies of the system.
    """

    cea_ex, vea_ex  = nonbond_energy_allneighbors(ff_,rtp_lat,ffa_) 
    cets,vets = nonbond_energy_intramol_correction(ff_,rtp_,ffa_)
    return cea_ex-cets, vea_ex-vets


@jit
def nonbond_energy_allneighbors(ff_ : ForceField,
                                rtp_: RTPCoord, # must be neghbors list type 
                                ffa_: ForceFieldAssignments # must include a neighbors-property 
                                ) -> f64:
    """
    Calculates the non-bonded energy of all neighbor interactions in the system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in RTP format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      tuple: Tuple containing the Coulomb and van der Waals energies of all neighbor interactions in the system.
    """

    def calc_coulomb_neighbors(r,neighbors_each,atomtypes,charges,dielectric_constant,ccutoff):

        ccoeff = jnp.where(jnp.sum(jnp.abs(neighbors_each[2:5]))==0,1,0.5)
        return calc_coulomb(r,atomtypes[neighbors_each[1]],atomtypes[neighbors_each[6]],\
                charges,dielectric_constant,ccoeff,ccutoff)

    v1cnei = vmap(calc_coulomb_neighbors,(   0,   0,None,None,None,None),0) 
    coulomb_energy_allpairs = v1cnei(rtp_.rall,ffa_.neighbors,ffa_.atomtypes,ff_.charges,ff_.dielectric_constant,ffa_.ccutoff)
    coulomb_energy_all = jnp.sum(jnp.sum(jnp.sum(jnp.sum(coulomb_energy_allpairs))))


    def calc_vdw_neighbors(r,neighbors_each,atomtypes,pairs,vcutoff):
        vcoeff = jnp.where(jnp.sum(jnp.abs(neighbors_each[2:5]))==0,1,0.5)
        return calc_vdw(r,atomtypes[neighbors_each[1]],atomtypes[neighbors_each[6]],pairs,vcoeff,vcutoff)

    v1vnei = vmap(calc_vdw_neighbors,(   0,   0,None,None,None),0) 
    vdw_energy_allpairs = v1vnei(rtp_.rall,ffa_.neighbors,ffa_.atomtypes,ff_.pairs,ffa_.vcutoff)
    vdw_energy_all = jnp.sum(jnp.sum(jnp.sum(jnp.sum(vdw_energy_allpairs))))

    return coulomb_energy_all, vdw_energy_all 

