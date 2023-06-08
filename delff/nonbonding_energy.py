from scipy import constants as const
from math import pi
from jax import vmap,jit,grad,jacfwd,lax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
from delff.objects import *
from delff.rtp import xyz2rtp,xyz2rtp_lattice

@jit
def nonbond_energy(ff_ : ForceField,
                   rtp_: RTPCoord,
                   ffa_: ForceFieldAssignments) -> f64:
    """Calculates the nonbonded energy of a system after correcting for intramolecular interactions.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      tuple: Total Coulomb energy and total van der Waals energy after corrections in Ha.
    """

    cea, vea  = nonbond_energy_allpairs(ff_,rtp_,ffa_)
    cets,vets = nonbond_energy_intramol_correction(ff_,rtp_,ffa_)
    return cea-cets,vea-vets


@jit
def calc_coulomb(r,atomtype0,atomtype1,charges,dielectric_constant,ccoeff,ccutoff):
    """Calculates the Coulomb interaction between two atom types.

    Arguments:
      r (float): Interatomic distance.
      atomtype0 (int): Atom type index of the first atom.
      atomtype1 (int): Atom type index of the second atom.
      charges (array): Array containing the charges of the atoms.
      dielectric_constant (float): Dielectric constant of the system.
      ccoeff (float): Coulomb coefficient.
      ccutoff (float): Coulomb interaction cutoff distance.

    Returns:
      float: Coulomb interaction energy in Ha.
    """
    charge0 = charges[atomtype0]
    charge1 = charges[atomtype1]
    r = jnp.where(r>1e-2,r,1e-2)
    ccoeff = charmm_Sfactor(r,ccutoff)*f64(r>1e-1)*ccoeff
    return jnp.where(ccoeff>1e-3,ccoeff*charge0*charge1/dielectric_constant/r,0) 

# Pair potential with the Lenard-Jones type function
@jit
def calc_vdw(r,atomtype0,atomtype1,pairs,vcoeff,vcutoff):
    """Calculates the van der Waals interaction between two atom types.

    Arguments:
      r (float): Interatomic distance.
      atomtype0 (int): Atom type index of the first atom.
      atomtype1 (int): Atom type index of the second atom.
      pairs (array): Array containing the epsilon and sigma parameters for the atoms.
      vcoeff (float): van der Waals coefficient.
      vcutoff (float): van der Waals interaction cutoff distance.

    Returns:
      float: van der Waals interaction energy in Ha.
    """

    r = jnp.where(r>1e-2,r,1e-2)
    vcoeff = f64(r<vcutoff)*f64(r>1e-1)*vcoeff
    #vcoeff = f64(r>1e-1)*vcoeff
    epsilon0 = pairs[atomtype0,0]
    sigma0   = pairs[atomtype0,1]
    epsilon1 = pairs[atomtype1,0]
    sigma1   = pairs[atomtype1,1]

    # arithmetic mixing for different atomtypes
    epsilon = jnp.sqrt(epsilon0*epsilon1)
    r0   = 0.5*(sigma0+sigma1)*jnp.power(2,1/6)  
    # calculated with n-exp form used in Gaussian
    r0_r_6 = jnp.power(r0/r,6)

    # normal type of LJ potential in https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # calculated with n-exp form used in Gaussian
    e_vdw = charmm_Sfactor(r,vcutoff)*vcoeff*epsilon*r0_r_6*(r0_r_6-2)  
    return e_vdw

@jit
def charmm_Sfactor(r,cutoff):
    """Calculates the CHARMM switching function value for a given distance and cutoff.

    Arguments:
      r (float): Interatomic distance.
      cutoff (float): Interaction cutoff distance.

    Returns:
      float: Value of the CHARMM switching function.
    """


    rin  = cutoff-2.0 # typically lj/charmm/coul/charmm[long] 8.0 10.0
    rout = cutoff

    def S(r, rin, rout):
        A=jnp.power(jnp.power(rout,2)-jnp.power(r,2),2)
        B=jnp.power(rout,2)+2*jnp.power(r,2)-3*jnp.power(rin,2)
        C=jnp.power(jnp.power(rout,2) - jnp.power(rin,2),3)
        return A*B/C

    Sin = jnp.where(r<=rin,1.0,0.0)
    Sboundary = jnp.where( jnp.logical_and(rin<r,r<rout), S(r,rin,rout), 0.0)
    return Sin+Sboundary


@jit
def nonbond_energy_allpairs(ff_ : ForceField,
                            rtp_: RTPCoord,
                            ffa_: ForceFieldAssignments) -> f64:

    """Calculates the nonbonded energy of all atom pairs in the system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      tuple: Total Coulomb energy and total van der Waals energy of all atom pairs in Ha.
    """

    dielectric_constant = ff_.dielectric_constant
    v1ceal = vmap(calc_coulomb,(   0,None,   0,None,None,None,None),0) # atom0
    v2ceal = vmap(v1ceal      ,(   0,None,None,None,None,None,None),0) # mol0
    v3ceal = vmap(v2ceal      ,(   0,   0,None,None,None,None,None),0) # atom1
    v4ceal = vmap(v3ceal      ,(   0,None,None,None,None,None,None),0) # mol1

    coulomb_energy_allpairs =                                      \
        v4ceal(rtp_.rall,ffa_.atomtypes,ffa_.atomtypes,ff_.charges,\
        dielectric_constant,1.,ffa_.ccutoff)

    coulomb_energy_all = jnp.sum(jnp.sum(jnp.sum(jnp.sum(coulomb_energy_allpairs))))

    v1veap = vmap(calc_vdw,(   0,None,   0,None,None,None),0) # atom0
    v2veap = vmap(v1veap  ,(   0,None,None,None,None,None),0) # mol0
    v3veap = vmap(v2veap  ,(   0,   0,None,None,None,None),0) # atom1
    v4veap = vmap(v3veap  ,(   0,None,None,None,None,None),0) # mol1

    vdw_energy_allpairs =                                          \
        v4veap(rtp_.rall,ffa_.atomtypes,ffa_.atomtypes,ff_.pairs,1.,ffa_.vcutoff)

    vdw_energy_all = jnp.sum(jnp.sum(jnp.sum(jnp.sum(vdw_energy_allpairs))))

    return 0.5*coulomb_energy_all, 0.5*vdw_energy_all
    # 0.5 for double count

@jit
def nonbond_energy_intramol_correction(ff_ : ForceField,           
                                       rtp_: RTPCoord,
                                       ffa_: ForceFieldAssignments) -> f64:
    """Calculates the nonbonded energy of a system after correcting for intramolecular interactions.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      rtp_ (RTPCoord): RTPCoord object containing the coordinates in rtp format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      tuple: Total Coulomb energy and total van der Waals energy after intramolecular corrections in Ha.
    """

    dielectric_constant = ff_.dielectric_constant
    nmol, natom, nmol1, natom1 = rtp_.rall.shape
    assert nmol == nmol1
    assert natom == natom1

    def diagonal_tsr4(tsr4,imol,natom):
        return jnp.reshape(tsr4[imol,:,imol,:,],(natom,natom))
    rintra = vmap(diagonal_tsr4,(None,0,None),0)(rtp_.rall,jnp.arange(nmol),natom)

    ccoeffmat = ffa_.adjmat012 + (1-ff_.cscale3)*ffa_.adjmat3
    vcoeffmat = ffa_.adjmat012 + (1-ff_.vscale3)*ffa_.adjmat3
    vcutoff = f64(ffa_.vcutoff)
    ccutoff = f64(ffa_.ccutoff)

    v1ceic = vmap(calc_coulomb,(   0,None,   0,None,None,   0,None),0)
    v2ceic = vmap(v1ceic      ,(   0,   0,None,None,None,   0,None),0)
    v3ceic = vmap(v2ceic      ,(   0,None,None,None,None,None,None),0)

    coulomb_energy_intramol_correction =                        \
        v3ceic(rintra,ffa_.atomtypes,ffa_.atomtypes,ff_.charges,\
        dielectric_constant,ccoeffmat,ccutoff)

    coulomb_energy_tosubstract = \
         jnp.sum(jnp.sum(jnp.sum(jnp.sum(coulomb_energy_intramol_correction))))


    v1veic = vmap(calc_vdw,(   0,None,   0,None,   0,None),0)
    v2veic = vmap(v1veic  ,(   0,   0,None,None,   0,None),0)
    v3veic = vmap(v2veic  ,(   0,None,None,None,None,None),0)

    vdw_energy_intramol_correction = \
        v3veic(rintra,ffa_.atomtypes,ffa_.atomtypes,ff_.pairs,vcoeffmat,vcutoff)

    vdw_energy_tosubstract = jnp.sum(jnp.sum(jnp.sum(vdw_energy_intramol_correction)))

    return 0.5*coulomb_energy_tosubstract, 0.5*vdw_energy_tosubstract


