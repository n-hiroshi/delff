from scipy import constants as const
from math import pi
from jax import vmap, value_and_grad, grad, lax, jvp, custom_jvp, jacfwd, jit,random
import jax.numpy as jnp
import jax.random as random
from jax.scipy.sparse.linalg import cg, gmres
from jax.scipy.linalg import solve
from scipy.optimize import minimize
import numpy as np
import delff.opt as opt
import delff.evalfunc as ef
from delff.util import logdamp, logdamp_np, jnp2np, np2jnp, convert_tsr4_to_mat2, printff, doreg, dounreg
from delff.objects import *
import copy
import optuna
import math

def metaopt(ff_ini: ForceField, 
            work: dict,
            maxiter: i32=100,
            algo='SDM',
            learning_rate=1.0,
            verbose=False):

    """
    This functioon optimized a force field using a given work specification.
    The meta-optimization process optimizes the parameters of the force field
    with respect to multiple tasks and multiple force field types.
    The function uses the scipy.minimize method with the SLSQP algorithm to optimize the force field parameters.

    Arguments:
        ff_ini (type: ForceField): The initial force field.
        work (type: dict): A dictionary that contains various parameters that define the optimization task.
        maxiter (type: i32, optional): The maximum number of iterations for the optimization. Defaults to 100.
        algo (type: str, optional): The optimization algorithm to use. Defaults to 'SDM'.
        learning_rate (type: float, optional): The learning rate for the optimization. Defaults to 1.0.
    
    Returns:
        The metaopt function return the optimezed force field (type: ForceField).
    
    """

    nbondtypes = ff_ini.bondtypes.shape[0]
    nangletypes = ff_ini.angletypes.shape[0]
    ndihedraltypes = ff_ini.dihedralks.shape[0]
    nimpropertypes = ff_ini.impropertypes.shape[0]
    npairs = ff_ini.pairs.shape[0]
    natoms = ff_ini.charges.shape[0]

    task_balance = work['task_balance']
    param_balance = work['param_balance']
    assert len(param_balance) == 6

    flag_opt = False

    print(nbondtypes,nangletypes,ndihedraltypes,nimpropertypes,npairs,natoms)

    ### (0) define a optmization step
    def vgh_np(vars_new_reg): 
        print("\n\n##### meta-minimization %d-th step #####"%len(resid_list))

        bondtypes,angletypes,dihedralks,impropertypes,pairs,charges = \
             np2jnp(vars_new_reg,nbondtypes,nangletypes,ndihedraltypes,\
                    nimpropertypes,npairs,natoms,vars_ini_full_reg,mask)

        # charges
        ff_new_reg  = ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,
                             ff_ini.dielectric_constant,ff_ini.vscale3,ff_ini.cscale3)

        printff(ff_new_reg)
        

        v,g =  value_and_grad(ef.L_sum,argnums=0)(ff_new_reg, work, algo) # workの中のalgoよりも別に引数にしたalgoが優先
        g = learning_rate*jnp2np(g.bondtypes,g.angletypes,g.dihedralks,g.impropertypes,g.pairs,g.charges,mask)

        norm_grad_list.append(jnp.linalg.norm(g))
        if np.any(np.isnan(g)): 
            switch2FDM=True
            print("\n Each elements of SDM.")
            _ =  ef.L_sum(ff_new_reg, work, 'SDM', verbose=True ) 
            print("\n Each elements of FDM.")
            _ =  ef.L_sum(ff_new_reg, work, 'FDM', verbose=True ) 
        elif norm_grad_list[1]*100.0 < norm_grad_list[-1]: 
            switch2FDM=False
            print("\n Each elements of SDM.")
            _ =  ef.L_sum(ff_new_reg, work, 'SDM', verbose=True ) 
            print("\n Each elements of FDM.")
            _ =  ef.L_sum(ff_new_reg, work, 'FDM', verbose=True ) 
        else: switch2FDM=False

        if switch2FDM:
            if algo == 'SDM':
                print("\n Switch from SDM to FDM.")

                bondtypes,angletypes,dihedralks,impropertypes,pairs,charges = \
                     np2jnp(vars_new_reg,nbondtypes,nangletypes,ndihedraltypes,\
                            nimpropertypes,npairs,natoms,vars_ini_full_reg,mask)
  
                # charges
                ff_new_reg  = ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,
                                     ff_ini.dielectric_constant,ff_ini.vscale3,ff_ini.cscale3)

                v,g =  value_and_grad(ef.L_sum,argnums=0)(ff_new_reg, work, 'FDM') # gがnanになったとときFDMに切り替え。
                g = learning_rate*jnp2np(g.bondtypes,g.angletypes,g.dihedralks,g.impropertypes,g.pairs,g.charges,mask)
        g[np.isnan(g)]=0.0
        print('\ngrad',g,'\n')

        resid_list.append(np.float64(v))
        ff_new = dounreg(ff_new_reg,reg)

        if algo in ['SDM','BO']:
            if min(resid_list[:-1]) > v:  ff_list.append(ff_new)
        elif algo == 'FDM':
            resid_SDM = ef.L_sum(ff_new_reg, work, 'SDM')
            resid_list_SDM.append(resid_SDM)
            print('\n# resid(SDM)',resid_SDM)
            if min(resid_list_SDM[:-1]) > resid_SDM:  ff_list.append(ff_new)

        print('\n# resid',v,'\n')

        return v,g

    ### (1) initalize
    print("\n\n#### Initialize the Optimization")
    maxiter_ncg = 0
    print('maxiter',maxiter)

    ## (1-1) set mask and bounds
    task0 = work['task_list'][0]
    ffa_  = task0['ffa']
    #total_charge = task0['total_charge']
    reg = work['reg']

    print("\n## Set mask")
    mask, chg_mask = set_mask(ff_ini, work)

    print("\n## Set bounds")
    bounds_full, ff_ini = set_bounds(ff_ini, work, mask)
    bounds_reg = doreg_and_mask_bounds(ff_ini, work, bounds_full, mask)

    ff_ini_reg = doreg(ff_ini,reg)
    vars_ini_full_reg = jnp2np(ff_ini_reg.bondtypes,ff_ini_reg.angletypes,ff_ini_reg.dihedralks,
                           ff_ini_reg.impropertypes,ff_ini_reg.pairs,ff_ini_reg.charges,np.ones(len(mask)))
    vars_ini_reg      = jnp2np(ff_ini_reg.bondtypes,ff_ini_reg.angletypes,ff_ini_reg.dihedralks,
                           ff_ini_reg.impropertypes,ff_ini_reg.pairs,ff_ini_reg.charges,mask)

    nvars = len(vars_ini_reg)
    print("num of vars",nvars)
    print("vars_ini_reg",vars_ini_reg)
    print()
    for var, bound in zip(vars_ini_reg, bounds_reg):
        print('var - bound:',var,bound)
        assert bound[0] <= var
        assert bound[1] >= var
        assert bound[0] < bound[1]+1e-8

    ## (1-2) neutral charge constraint 
    print("\n## Set charge contraint")
    def const_neutral(x0,chg_mask=chg_mask,atomtypes=ffa_.atomtypes,total_charge=0.0,reg=reg):
        assert len(x0)==len(chg_mask)
        charges_atomtype = x0[chg_mask==1]

        if len(charges_atomtype) ==0: return 0.0 # no charge optimization
        else:
            charges = jnp.array([charges_atomtype[atomtype] for atomtype in atomtypes])
        return np.sum(charges)#-total_charge/reg[11]

    if jnp.sum(chg_mask)>0:
        constraints={'type': 'eq',
                     'fun' : const_neutral}, 
    else:
        constraints=None

   
  
    const_neutral(vars_ini_reg,chg_mask=chg_mask,atomtypes=ffa_.atomtypes,total_charge=0.0,reg=reg)
    print(constraints)
    

    ## (1-3)  calc the initial state
    resid = ef.L_sum(ff_ini_reg, work, algo)
    print('\n#init_resid',resid)
    ff_list=[ff_ini]
    resid_list = [np.float64(resid)]
    norm_grad_list = [0.0]
    if algo == 'FDM':
        resid_SDM = ef.L_sum(ff_ini_reg, work, 'SDM')
        resid_list_SDM = [np.float64(resid_SDM)]
        print('\n#init_resid(SDM)',resid_SDM)

    ### (3) optimiztion
    if algo in ['SDM','FDM']:


        minimize_res = minimize(vgh_np,
                             x0=vars_ini_reg,
                             bounds=bounds_reg,
                             constraints=constraints,
                             #constraints={'type': 'eq',
                             #             'fun' : const_neutral}, 
                             # ref: https://qiita.com/imaizume/items/44896c8e1dd0bcbacdd5
                             method='SLSQP',#'Newton-CG',#'TNC',#"TNC",#"SLSQP""TNC"
                             jac=True,
                             tol=1e-10,
                             options={
                                 "maxiter": maxiter,
                                 "disp": True,
                                 "iprint": 2,
                                 #"gtol": 1e-10,
                                 "ftol": 1e-10 } )

        print(minimize_res)
        var = jnp.asarray(minimize_res.x)

        var_opt = var
        print("\n## meta-optimised parameters")
        print(var_opt)
        ### output results
        ff_opt = ff_list[-1]

    elif algo=='BO':
        ## optuna-opt 
        import optuna
            # charge neutral constraint
        def neutralize(vars_new_reg, chg_mask, atomtypes):
            sum_charges = const_neutral(vars_new_reg, chg_mask=chg_mask, atomtypes=atomtypes)
            charges_atomtype = vars_new_reg[chg_mask==1]
            charges = jnp.array([charges_atomtype[atomtype] for atomtype in ffa_.atomtypes])
            #charges -= sum_charges/len(charges)
            delcharges = sum_charges/len(charges)
            charges_atomtype -= delcharges
            vars_new_reg[chg_mask==1] = charges_atomtype
            sum_charges = const_neutral(vars_new_reg, chg_mask=chg_mask, atomtypes=ffa_.atomtypes)
            assert sum_charges < 1e-6
            return vars_new_reg

 
        def objective(trial):
    
            trial_varvec = np.zeros(len(bounds_reg))
            for ivar,bound in enumerate(bounds_reg):
                trial_varvec[ivar] = trial.suggest_float('var%d'%ivar,bound[0],bound[1])
                #if mask[ivar] ==1:
                #else: 
                #    trial_varvec[ivar] =  varvec_ini[ivar]

            # covert from vars_new to FF-types
            vars_new_reg = trial_varvec
            vars_new_reg = neutralize(vars_new_reg, chg_mask, ffa_.atomtypes)
            
            bondtypes,angletypes,dihedralks,impropertypes,pairs,charges = \
                 np2jnp(vars_new_reg,nbondtypes,nangletypes,ndihedraltypes,\
                        nimpropertypes,npairs,natoms,vars_ini_full_reg,mask)
            ff_new_reg  = ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,
                                 ff_ini.dielectric_constant,ff_ini.vscale3,ff_ini.cscale3)

    
            resid =  ef.L_sum(ff_new_reg, work, algo)
            return resid

        study = optuna.create_study()
        study.optimize(objective, n_trials=maxiter)
        #study.best_params  # E.g. {'x': 2.002108042}
        df_history = study.trials_dataframe()
        resid_list_from_st1 = list(df_history['value'])
        resid_list += resid_list_from_st1
        print(resid_list)

        ## collect best params
        opted_varvec = np.zeros(len(bounds_reg))
        for ivar in range(len(bounds_reg)):
                opted_varvec[ivar] = study.best_params['var%d'%ivar]

        # covert from vars_new to FF-types
        vars_new_reg = opted_varvec
        vars_new_reg = neutralize(vars_new_reg, chg_mask, ffa_.atomtypes)

        bondtypes,angletypes,dihedralks,impropertypes,pairs,charges = \
             np2jnp(vars_new_reg,nbondtypes,nangletypes,ndihedraltypes,\
                    nimpropertypes,npairs,natoms,vars_ini_full_reg,mask)
        ff_opt_reg  = ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges,
                             ff_ini.dielectric_constant,ff_ini.vscale3,ff_ini.cscale3)
        print('ff_opt_reg')
        print(ff_opt_reg)
        ff_opt = dounreg(ff_opt_reg,reg)


    # (4) output results
    print("\n### output results")
    printff(ff_opt)
    print("total-charges",jnp.sum(ff_opt.charges))
    print("change_bondtypes",ff_opt.bondtypes-ff_ini.bondtypes)
    print("change_angletypes",ff_opt.angletypes-ff_ini.angletypes)
    print("change_dihedraltypes",ff_opt.dihedralks-ff_ini.dihedralks)
    print("change_impropertypes",ff_opt.impropertypes-ff_ini.impropertypes)
    print("change_pairs",ff_opt.pairs-ff_ini.pairs)
    print("change_charges",ff_opt.charges-ff_ini.charges)

    return ff_opt


def set_mask(ff_ini: ForceField, work: dict):
    """
    This function generates a mask array based on the input force field and the given work type.
    The generated mask array is used to zero-out certain force field parameters during optimization.

    Arguments:
        ff_ini: An instance of the ForceField class representing the initial force field.
        work (type: dict): A dictionary that contains various parameters that define the optimization task.
    Returns:
        mask: A 1D numpy array of integers representing the mask for the force field parameters. 
        The elements of the array are either 0 or 1, indicating whether or not the corresponding parameter
        should be regularized.
        chg_mask: A 1D numpy array of integers representing the mask for the atomic charges in the force field.
        This mask is generated in the same way as the mask array, but only includes the mask for the charge parameters.
    """

    param_balance = work['param_balance']
    task0 = work['task_list'][0]
    ffa_  = task0['ffa']
        
    nbondtypes = ff_ini.bondtypes.shape[0]
    nangletypes = ff_ini.angletypes.shape[0]
    ndihedraltypes = ff_ini.dihedralks.shape[0]
    nimpropertypes = ff_ini.impropertypes.shape[0]
    npairs = ff_ini.pairs.shape[0]
    natoms = ff_ini.charges.shape[0]

    mask_bond      = f64(param_balance[0]>0)
    mask_angle     = f64(param_balance[1]>0)
    mask_dihedrals = f64(param_balance[2]>0)
    mask_improper  = f64(param_balance[3]>0)
    mask_pair      = f64(param_balance[4]>0)
    mask_charge    = f64(param_balance[5]>0)
    # print(len(mask_dihedrals))

    # No Regularization
    dihedralphis_ravel = ffa_.dihedralphis.ravel()
    dihedralks_ravel = ff_ini.dihedralks.ravel() 
    print('\n# dihedralks_ravel:',dihedralks_ravel)
    print('Now using all four cosine elements for each dihdral')
    #for i, a in enumerate(ffa_.dihedralmasks.ravel()):
    #    mask_dihedrals[i]   = a*mask_dihedrals[i]

    mask  = [mask_bond,mask_bond]*nbondtypes +\
            [mask_angle,mask_angle]*nangletypes +\
            [mask_dihedrals]*4*ndihedraltypes +\
            [mask_improper]*nimpropertypes*3 +\
            [mask_pair]*npairs*2 + \
            [mask_charge]*natoms

    mask  = np.asarray(mask,np.int32)
    print('param_balance',param_balance)
    print('mask',mask,'len(mask)',len(mask))

    chg_mask  = [False,False]*nbondtypes +\
            [False,False]*nangletypes +\
            [False]*4*ndihedraltypes +\
            [False]*nimpropertypes*3 +\
            [False]*npairs*2 + \
            [mask_charge]*natoms
    chg_mask  = np.asarray(chg_mask,np.int32)

    chg_mask = chg_mask[mask>0]

    print('chg_mask',chg_mask,'len(chg_mask)',len(chg_mask))
    print()
    return mask, chg_mask


def set_bounds(ff_ini: ForceField, work: dict, mask: Array):
    """
    This function defines a function set_bounds() which sets bounds on force field parameters based on given inputs.
    Arguments: 
        ff_ini (ForceField): The initial force field object.
        work (dict): A dictionary containing additional information about the force field.
        mask (Array): A Boolean array indicating whether to apply bounds on a given force field parameter.
    Returns:
        bounds (dict): A dictionary containg boundary values of each force field parameters.
        ff_ini (ForceField): The modified iniital force field object.
            The paramters for pairs (vdw-parameters) are adaptively modifled for avoiding errors.
    """
    task0 = work['task_list'][0]
    ffa_  = task0['ffa']
    vratio = 1.00
    kdihedmax = 5.0
    drad   = 180.0
    dpair0 = 1.0
    dpair1 = 4.0
    dcharge=0.5

    nbondtypes = ff_ini.bondtypes.shape[0]
    nangletypes = ff_ini.angletypes.shape[0]
    ndihedraltypes = ff_ini.dihedralks.shape[0]
    nimpropertypes = ff_ini.impropertypes.shape[0]
    npairs = ff_ini.pairs.shape[0]
    natoms = ff_ini.charges.shape[0]

    bounds =[]
    print('bounds')

    for i in range(nbondtypes):
        add_bound0  = ((1-vratio)*np.float64(ff_ini.bondtypes[i,0]),
                       (1+vratio)*np.float64(ff_ini.bondtypes[i,0]))
        add_bound1  = ((1-vratio)*np.float64(ff_ini.bondtypes[i,1]),
                       (1+vratio)*np.float64(ff_ini.bondtypes[i,1]))
        bounds += [add_bound0,add_bound1]

    for i in range(nangletypes):
        add_bound0 = ((1-vratio)*np.float64(ff_ini.angletypes[i,0]),
                      (1+vratio)*np.float64(ff_ini.angletypes[i,0]))
        add_bound1 = (np.float64(ff_ini.angletypes[i,1])-drad,
                      np.float64(ff_ini.angletypes[i,1]+drad))
        bounds += [add_bound0,add_bound1]

    for i in range(ndihedraltypes):
        for j in range(4):
            add_bound0 = (-kdihedmax,kdihedmax)
            bounds += [add_bound0]

    for i in range(nimpropertypes):
        add_bounds = []
        add_bounds += [((1-vratio)*np.float64(ff_ini.impropertypes[i,0]),
                        (1+vratio)*np.float64(ff_ini.impropertypes[i,0]))]
        add_bounds += [(np.float64(ff_ini.impropertypes[i,1])-drad,
                        np.float64(ff_ini.impropertypes[i,1])+drad)]
        add_bounds += [((1-vratio)*np.float64(ff_ini.impropertypes[i,2]),
                        (1+vratio)*np.float64(ff_ini.impropertypes[i,2]))]
        bounds += add_bounds

    for i in range(npairs):
        add_bounds = []
        if ff_ini.pairs[i,0] > 0.01:
            add_bounds += [(0.001,
                            (1+vratio)*np.float64(ff_ini.pairs[i,0]))]
        else:
            add_bounds += [(0.001,dpair0)] #The exact 0.0 of epsilon makes errors in grad operation

            # avoid 'Inequality constraints incompatible' in metaopt
            if type(ff_ini.pairs) is np.ndarray: ff_ini.pairs[i,0]=0.01
            else:
                pairs = ff_ini.pairs.at[i,0].set(0.01)
                ff_ini = update(ff_ini,pairs=pairs)

        if ff_ini.pairs[i,1] > 1.0:
            add_bounds += [(0.05,
                           (1+vratio)*np.float64(ff_ini.pairs[i,1]))]
        else:
            add_bounds += [(0.05,dpair1)]
            # avoid 'Inequality constraints incompatible' in metaopt
            if type(ff_ini.pairs) is np.ndarray: ff_ini.pairs[i,1]=1.0
            else:
                pairs = ff_ini.pairs.at[i,1].set(1.0)
                ff_ini = update(ff_ini,pairs=pairs)
        bounds += add_bounds

    for i in range(natoms):
        add_bounds = []
        add_bounds += [(-dcharge+np.float64(ff_ini.charges[i]),dcharge+np.float64(ff_ini.charges[i]))]
        bounds += add_bounds

    bounds_full = bounds
    
    return bounds_full, ff_ini

def doreg_and_mask_bounds(ff_ini, work, bounds_full, mask):
    """
    This function takes a force field object, a dictionary of work, an array of full bounds,
    and an array of mask as input. It generates a new array of bounds
    by dividing each element of the full bounds array by the corresponding element of the reg array,
    which is extracted from the work dictionary.
    It then filters the bounds array using the mask array, and returns the resulting bounds array.

    Arguments:
        ff_ini: a ForceField object.
        work: a dictionary of work.
        bounds_full: an array of full bounds.
        mask: an array of mask.
    Returns:
        bounds_reg: a filtered array of bounds.
    """
    reg = np.array(work['reg'])
    types = [
        ('bond', ff_ini.bondtypes, [0,1]),
        ('angle', ff_ini.angletypes, [2,3]),
        ('dihedral', ff_ini.dihedralks, [4,4,4,4]),
        ('improper', ff_ini.impropertypes, [6,7,8]),
        ('pair', ff_ini.pairs, [9,10]),
        ('charge', ff_ini.charges, [11])
    ]
    l = 0
    bounds_reg = []
    for type_name, type_data, indeces in types:
        for i in range(type_data.shape[0]):
            for k in indeces:
                bounds_reg.append(np.array(bounds_full[len(bounds_reg)]) / reg[k])
    bounds_reg = [bounds_reg[i] for i in range(len(bounds_reg)) if mask[i] > 0]
    return bounds_reg







