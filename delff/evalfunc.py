from scipy import constants as const
from math import pi
from functools import partial
from jax import vmap, value_and_grad, grad, hessian, lax, jvp, custom_jvp, jacfwd, jit,random
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
from delff import energy, opt, opt_lat, rtp, util
from delff.objects import *
from delff.rtp import xyz2rtp
import copy
from jax.experimental.host_callback import id_print

def L_sum(ff_reg : ForceField, work: dict, algo: str, verbose=False ) -> f64:
    """Calculates the sum of residuals for different types of tasks: optimization, optimization with lattice, and potential energy surface.

    Arguments:
      ff_reg (ForceField): ForceField object containing the parameters for the force field.
      work (dict): A dictionary containing various task parameters and balances.
      algo (str): The algorithm to be used, either 'SDM', 'BO', or 'FDM'.
      verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
      f64: The sum of residuals.
    """

    task_list     = work['task_list']
    task_balance  = work['task_balance']
    param_balance = work['param_balance']
    reg           = work['reg']

    ff_ = util.dounreg(ff_reg,reg)

    sum_resid=0.0
    for itask,task in enumerate(task_list):
 
        if task['type'] == "opt":
            ffa_ = task['ffa']
            sys_ = task['sys']

            if ffa_.nmolvec[0] ==1:

                if algo in ['SDM','BO']:
                    scale = task_balance[itask]*task['scale_structure']
                    resid = scale*L_sys(ff_,sys_,ffa_,reg)[0]
                elif algo == 'FDM':
                    scale = task_balance[itask]*task['scale_force']
                    resid = scale*L_force(ff_,sys_,ffa_)
                if verbose: print('Resid(%d) %s: %10.8f'%(itask,task['type']+algo, resid))

        elif task['type'] == "opt_lat":
            ffa_ = task['ffa']
            sys_ = task['sys']
            Ulattice = task['Ulattice']
            if algo in ['SDM','BO']:
                scale = task_balance[itask][0]*task['scale_structure']
                resid_sys =scale*L_sys_lat(ff_,sys_,ffa_,reg)
            elif algo == 'FDM':
                scale = task_balance[itask][0]*task['scale_force']
                resid_sys =scale*L_force(ff_,sys_,ffa_)
            if verbose: print('Resid_sys(%d) %s: %10.8f'%(itask,task['type']+algo, resid_sys))
            
            # Ulattice matching
            scale = task_balance[itask][1]*task['scale_energy']
            Ulattice = task['Ulattice']
            sys_tgt_monomer = task_list[0]['sys']
            assert sys_tgt_monomer.coord.shape[0] == 1 # check if sys_tgt_monomer is a monomer

            if algo in ['SDM','BO']:
                resid_ene = scale*L_Ulattice(ff_,sys_,ffa_,sys_tgt_monomer,Ulattice)
            elif algo == 'FDM':
                resid_ene = scale*L_Ulattice_tgt(ff_,sys_,ffa_,sys_tgt_monomer,Ulattice)

            if verbose: print('Resid_ene(%d) %s: %10.8f'%(itask,task['type']+algo, resid_ene))
            resid = resid_ene + resid_sys

        elif task['type'] == "pes":
            ffa_ = task['ffa']
            sys_pes = task['sys_pes']
            energies  = task['energies']
            scale = task_balance[itask]*task['scale_energy']
            resid = scale*L_pes(ff_,sys_pes,ffa_,energies)
            if verbose: print('Resid(%d) %s: %10.8f'%(itask,task['type'], resid))

        sum_resid += resid

    charges = ff_.charges[ffa_.atomtypes]
    check_neutral = jnp.sum(charges)
    if verbose: print('check sum of charges: %10.8f'%check_neutral)

    return sum_resid

@jit
def L_sys(ff_: ForceField, 
          sys_tgt: System,
          ffa_: ForceFieldAssignments,
          reg: Array) -> f64:
    """Computes the residual between the system's optimized and target RTP coordinates.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_tgt (System): Target System object containing the coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      reg (Array): An array containing the regularization parameters.

    Returns:
      f64: The residual.
    """

    assert len(ffa_.nmolvec)==1
    sys_opt = opt.opt_sys(ff_,sys_tgt,ffa_)

    rtp_opt = rtp.xyz2rtp(sys_opt, ffa_)
    target_rtp = rtp.xyz2rtp(sys_tgt, ffa_)

    diff_rs     = 1/reg[1]*(rtp_opt.rs-target_rtp.rs).ravel()
    diff_thetas = 1/reg[3]*(rtp_opt.thetas-target_rtp.thetas).ravel()
    diff_phids  = 1/reg[5]*(rtp_opt.phids-target_rtp.phids).ravel()
    diff_phiis  = 1/reg[8]*(rtp_opt.phiis-target_rtp.phiis).ravel()

    diff_vec = jnp.concatenate((diff_rs,diff_thetas,diff_phids,diff_phiis))
    resid = jnp.dot(diff_vec,diff_vec)

    return resid, sys_opt

@jit
def L_sys_lat(ff_: ForceField,
              sys_tgt: System,
              ffa_: ForceFieldAssignments,
              reg: Array,
              coeff_lattice: f64=10.0) -> f64:
    """Computes the residuals between the system's optimized and target lattice and coordinates.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_tgt (System): Target System object containing the coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      reg (Array): An array containing the regularization parameters.
      coeff_lattice (f64, optional): Coefficient for the lattice residual. Defaults to 10.0.

    Returns:
      f64: The combined residual of lattice and coordinates.
    """
    
    sys_opt = opt_lat.opt_lat_sys(ff_,sys_tgt,ffa_)

    diff_lattice = sys_opt.lattice - sys_tgt.lattice
    dim1_diff_lattice = 1/reg[10]*diff_lattice.ravel()
    resid0 = jnp.dot(dim1_diff_lattice,dim1_diff_lattice)

    diff_coord = sys_opt.coord - sys_tgt.coord
    dim1_diff_coord = diff_coord.ravel()
    resid1 = jnp.dot(dim1_diff_coord,dim1_diff_coord)

    id_print(jnp.sqrt(jnp.mean(jnp.mean(jnp.linalg.norm(diff_coord,axis=2)**2))))

    return coeff_lattice*resid0 + resid1


@jit
def L_Ulattice(ff_: ForceField, 
               sys_tgt_lat: System,
               ffa_: ForceFieldAssignments,
               sys_tgt_monomer: System,
               U_lattice: f64) -> f64:
    """Computes the residual between the lattice energy calculated with the force field and the target lattice energy at the optimized structures with the ff_.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_tgt_lat (System): Target System object for the lattice containing the coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      sys_tgt_monomer (System): Target System object for the monomer containing the coordinates in xyz format.
      U_lattice (f64): Target lattice energy.

    Returns:
      f64: The energy residual.
    """

    nmol,_,__ = sys_tgt_lat.coord.shape
    sys_opt_lat = opt_lat.opt_lat_sys(ff_,sys_tgt_lat,ffa_)
    sys_opt_monomer = opt.opt_sys(ff_,sys_tgt_monomer,ffa_)

    Ulatt_ff = energy.energy_coord(ff_,sys_opt_lat,ffa_)/nmol  \
             - energy.energy_coord(ff_,sys_opt_monomer,ffa_)
    return (Ulatt_ff-U_lattice)**2


@jit
def L_Ulattice_tgt(ff_: ForceField, 
               sys_tgt_lat: System,
               ffa_: ForceFieldAssignments,
               sys_tgt_monomer: System,
               U_lattice: f64) -> f64:
    """Computes the residual between the lattice energy calculated with the force field and the target lattice energy at the target lattice structures.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_tgt_lat (System): Target System object for the lattice containing the coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      sys_tgt_monomer (System): Target System object for the monomer containing the coordinates in xyz format.
      U_lattice (f64): Target lattice energy.

    Returns:
      f64: The energy residual.
    """


    nmol,_,__ = sys_tgt_lat.coord.shape

    Ulatt_ff = energy.energy_coord(ff_,sys_tgt_lat,ffa_)/nmol  \
             - energy.energy_coord(ff_,sys_tgt_monomer,ffa_)
    return (Ulatt_ff-U_lattice)**2



@jit
def L_force(ff_: ForceField, 
             sys_tgt: System, 
             ffa_: ForceFieldAssignments) -> f64:
             #reg : Array) -> f64:
    """Computes the square of the norm of the gradient of the total energy with respect to the coordinates of the system.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_tgt (System): Target System object containing the coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.

    Returns:
      f64: The square of the norm of the gradient.
    """

    etot,grad_etot = value_and_grad(energy.energy_coord,argnums=1) \
                                   (ff_,sys_tgt, ffa_) 
    dim1_grad_etot = grad_etot.coord.ravel() # TAKE CARE for introcucing PBC
    resid = jnp.dot(dim1_grad_etot,dim1_grad_etot)
    return resid

@jit
def L_pes(ff_: ForceField, 
          sys_pes: System,
          ffa_: ForceFieldAssignments,
          energies: Array) -> f64:
    """Computes the residual of the potential energy surface (PES) based on a Boltzmann-weighted square difference between reference and computed energies.

    Arguments:
      ff_ (ForceField): ForceField object containing the parameters for the force field.
      sys_pes (System): System object containing the PES coordinates in xyz format.
      ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
      energies (Array): An array of reference energies for each structure in the PES.

    Returns:
      f64: The PES residual.
    """


    min_g16_energy = jnp.min(jnp.array(energies))
    imin_structure = jnp.argmin(jnp.array(energies))
    energy_ref_vec = jnp.array(energies)-min_g16_energy
    energy_delff_vec = jnp.zeros_like(energy_ref_vec)

    nstr, nmol, natom, _ = sys_pes.coord.shape

    def energy_coord_array_input(ff_,coord_, ffa_):
        sys_ = System(coord_)
        return energy.energy_coord(ff_,sys_,ffa_)
    energy_delff_vec = vmap(energy_coord_array_input,(None,0,None),0)(ff_, sys_pes.coord, ffa_)


    Kb = 8.617333262E-5 # [eV/K]
    KbT = 2000.0*Kb # [eV]
    KbT = KbT/27.2114 # [Hartree]

    energy_delff_vec_zero_corrected = (energy_delff_vec - energy_delff_vec[imin_structure])

    energy_mat = jnp.stack((energy_ref_vec,energy_delff_vec_zero_corrected),axis=0)
    energy_min = jnp.min(energy_mat,axis=0)
    prob_boltzmann_2 = jnp.exp(-energy_min/KbT/2) 
    resid_vec = energy_ref_vec - energy_delff_vec_zero_corrected
    resid_vec = jnp.multiply(prob_boltzmann_2,resid_vec)

    resid = jnp.dot(resid_vec,resid_vec)

    # log damp
    resid = jnp.where(resid<1.0,resid,1.0+jnp.log(resid))
    return resid


