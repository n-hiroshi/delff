import pytest,warnings
import os,sys,pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from jax import numpy as jnp, lax, vmap, jacfwd, grad, nn as jnn, jit

# delff - original libraries https://github.com/n-hiroshi/delff.git
import delff.lammpsff as lff
import delff.gaussianff as gff
import delff.evalfunc as ef
from delff.objects import *
from delff.params_ewald import *
from  delff import energy,opt,opt_lat,metaopt,rtp,util
#from delff.gaussianhandler import GaussianHandler


def prepare(root,
            name,
            idx,
            task_balance,
            taskids,
            monomer_gjf,
            atomtypesets,
            in_settings_file,
            data_file,
            path_ff_load = "",
            #param_balance = jnp.asarray([1,1,1,0,1,0]),
            param_balance = jnp.asarray([1,1,1,0,1,1]),
            lat_gjf = "",
            Ulattice = 0.0,
            pes_dirs = [],
            dielectric_constant = 3.0,
            ewald=True,
            alpha_ewald=0.5,
            nkmaxs = jnp.asarray([3,3,3])
            ):
    '''
    The function prepare is used to prepare the input for the force field optimization. 
    It takes multiple arguments and returns a dictionary containing relevant information.
    
    Arguments:
        root (str): the directory where the optimization is run
        name (str): the name of the optimization
        idx (int): an integer index of the optimization
        task_balance (numpy.ndarray): an array of integers
        specifying the weights of each task
        taskids (list): a list of integers representing the ids of tasks to perform
        monomer_gjf (str): the path of the Gaussian input file for the monomer
        atomtypesets (list): a list of sets containing the names of the atoms 
        in each type of molecule
        in_settings_file (str): the path of the lammps-settings file for the force field
        data_file (str): the path of the lammps-data file for the force field
        path_ff_load (str): the path of the file to load the force field parameters 
        from (optional, default="")
        param_balance (numpy.ndarray): an array of integers representing the weights 
        of different types of parameters (optional, default=[1,1,1,0,1,1])
        lat_gjf (str): the path of the Gaussian input file
        for the crystal (optional, default="")
        Ulattice (float): the lattice energy of the crystal
        in Hartree (optional, default=0.0)
        pes_dirs (list): a list of directories containing PES data (optional, default=[])
        dielectric_constant (float): the dielectric constant of the solvent
        (optional, default=3.0)
        ewald (bool): whether or not to use Ewald summation
        for electrostatics (optional, default=True)
        alpha_ewald (float): the alpha parameter for Ewald summation (optional, default=0.5)
        nkmaxs (numpy.ndarray): an array of integers representing the number of k vectors
        to use in Ewald summation (optional, default=[3,3,3])

    Returns:
        work (dict): a dictionary containing the information needed for the optimization,
        including the following keys:
        root (str): the directory where the optimization is run
        name (str): the name of the optimization
        idx (int): an integer index of the optimization
        param_balance (numpy.ndarray): an array of integers 
        representing the weights of different types of parameters
        reg (numpy.ndarray): an array of regularization 
        factors for different types of parameters
        task_balance (numpy.ndarray): an array of integers
        specifying the weights of each task
        all_task_list (list): a list of dictionaries,
            each containing the following keys:
                sys (parmed.Structure): the structure for the task
                atomtypelabels (list): a list of strings
                representing the atom types in the task
                scale_structure (float): a scaling factor for the structure energy
                scale_force (float): a scaling factor for the force
                scale_energy (float): a scaling factor for the energy
                type (str): a string representing the type of task 
                ('opt','opt_lat', and 'pes')
    '''
    print('\n\n\n############### PREPERATION ###############')


    work={}
    work['root']=root
    work['name']=name
    work['idx'] = idx
    work['param_balance'] = param_balance
    reg = jnp.asarray([100,0.1,100,20,1,20,  # bond,angle,dihed
                       1,5,20,0.1,1,0.1]) # improper,vdw,charge
    work['reg'] = reg # bond,angle,dihed
    coeff_E = kcalM2Ha**2
    work['task_balance'] = task_balance
    work['root'] = root
    all_task_list=[]


    ### (1) monomer
    print('\n\n### Monomer Structure Matching')
    task={}
    natom = len([v for eachset in atomtypesets for v in eachset ])

    #sys_tgt_monomer = gff.get_sys_from_gjf(monomer_gjf)   ### 230307 refactoring
    #natom = sys_tgt_monomer.coord.shape[1] # nmol x natom x 3dim### 230307 refactoring
    ff_ini,sys_ini,ffa_,atomtypelabels = lff.read_system(in_settings_file,data_file,\
            atomtypesets=atomtypesets) ### 230307 refactoring
    sys_tgt_monomer = sys_ini  ### 230307 refactoring


    ff_ini = update(ff_ini,dielectric_constant=dielectric_constant)

    ## charge damping
    if len(path_ff_load) ==0:
        ff_ini = update(ff_ini,charges=ff_ini.charges)

    if len(path_ff_load) >=1: 
        print('%s loaded to ff_ini'%path_ff_load)
        with open(path_ff_load, 'rb') as f: ff_ini = pickle.load(f)


    util.printff(ff_ini)
    #print('\n# ff_ini\n',ff_ini)
    #print('\n# ffa_\n',ffa_)
    print('\n# atomtypelabels',atomtypelabels)
    task['atomtypelabels'] = atomtypelabels

    # sys
    task['sys']=sys_tgt_monomer

    # ffa
    task['ffa'] = ffa_

    # scale
    task['scale_structure'] = 1.0/natom#0.01#/np.float64(Lval)
    task['scale_force']=coeff_E/natom/20.0**2

    # type
    task['type'] = 'opt'
    all_task_list.append(task)


    ### (2) crtystal
    print('\n\n### Crystal Structure Matching')
    task={}
 
    # sys
    sys_tgt_lat = gff.get_sys_from_gjf(lat_gjf)
    task['sys']=sys_tgt_lat                    
    print('\n# target crystal lattice\n',sys_tgt_lat.lattice)
    #print('\n# target crystal coord',sys_tgt_lat.coord)
    print('\n# target crystal coord shape\n',sys_tgt_lat.coord.shape)
                                            
    ## coeff_formation_energy               
    task['Ulattice'] = Ulattice #[Ha]
                                        
    # ffa                               
    nmol,natom_lat,_ = sys_tgt_lat.coord.shape
    assert natom == natom_lat
    neighbors,latidx = search_neighbors(sys_tgt_lat,ccutoff=ffa_.ccutoff)
    ffa_lat = update(ffa_,nmolvec=jnp.array([nmol]),natomvec=jnp.array([natom]),\
            neighbors=neighbors,latidx=latidx)
    rall = rtp.alldists_neighbors(sys_tgt_lat,ffa_lat)
    assert jnp.max(rall) < ffa_.ccutoff + 2.0
                                        
    # ewald
    if ewald: 
        ffa_lat = update(ffa_lat,alpha_ewald=alpha_ewald,nkmaxs=nkmaxs)
        ffa_lat = define_ewald_params(sys_tgt_lat,ffa_lat)
    task['ffa'] = ffa_lat          

    # scale                             
    task['scale_structure']=1.0/natom/nmol
    task['scale_force']=coeff_E/natom/nmol/20.0**2      
    task['scale_energy']=0.25*coeff_E/natom

 
    # type
    task['type'] = 'opt_lat'
    all_task_list.append(task)
    print(len(all_task_list))


    ### (3) pes0
    print('\n\n### PES Matching')
    pes_list = gff.get_ref_sys_and_energies(pes_dirs,natom)

    #print(len(pes_list))
    for ipes, pes in enumerate(pes_list):
        task={}
        sys_pes = pes_list[ipes][0]
        energies  = pes_list[ipes][1]
        Lval= ef.L_pes(ff_ini,sys_pes,ffa_,energies)
        task['ffa'] = ffa_
        task['sys_pes'] = sys_pes
        task['energies'] = energies
        task['scale_energy'] = 0.25*coeff_E/natom/len(energies)
        task['type'] = 'pes'
        all_task_list.append(task)


    ## (4) set task_list to work dict
    task_list=[]
    taskids_inv = []
    for taskid in taskids:
        task_list.append(all_task_list[taskid])
    work['task_list'] = task_list
    work['taskids'] = taskids

    return ff_ini, work


def run(ff_ini,work,algo,maxiter=100,pes_dirs=[]):
    """
    This function is the main function for optimizing force fields. 
    It takes an initial force field, a work dictionary, an optimization algorithm,
    and other optional arguments such as maximum iterations
    and potential energy surface directories, and returns the optimized force field.

    Arguments:
        ff_ini: an initial force field
        work: a dictionary containing information about the system 
        and the optimization parameters
        algo: a string indicating the optimization algorithm to use (SDM, FDM, or BO)
        maxiter: an integer indicating the maximum number
        of iterations for the optimization algorithm (default is 100)
        pes_dirs: a list of potential energy surface directories 
        for plotting (default is an empty list)

    Returns:
       ff_opt: the optimized force field

    """


    root = work['root']
    idx  = work['idx']
    reg = work['reg']

    ## (1) calc init L_sum with algo=SD mode (even FD optimization mode)
    print('\n\n\n############### INITIAL STATE ###############')
    ff_ini_reg = util.doreg(ff_ini,reg)
    Lval_sum = ef.L_sum(ff_ini_reg,  work, algo, verbose=True)
    print('# INITIAL L_sum(%s): %10.8f'%(algo,Lval_sum))
    if algo == 'FDM':
       Lval_sum = ef.L_sum(ff_ini_reg,  work, 'SDM', verbose=True)
       print('# INITIAL L_sum(SDM): %10.8f'%Lval_sum)


    ## (2) output initial state with algo=SD mode (even FD optimization mode)
    save_to_gjf_lmp(ff_ini,work,'ini')
    
    ## (3) METAOPT with algo=SD/FD/BO mode optimization
    print('\n\n\n############### OPTMIZATION ###############')
    ff_opt = metaopt.metaopt(ff_ini, work,maxiter=maxiter,algo=algo)
    #ff_opt = ff_ini
    path_ff_save = root + "ff_opt%02d.pickle"%idx
    with open(path_ff_save, 'wb') as f: pickle.dump(ff_opt,f)

    ## (4) calc final L_sum with algo=SD mode (even FD optimization mode)
    print('\n\n\n############### FINAL STATE ###############')
    ff_opt_reg = util.doreg(ff_opt,reg)
    Lval_sum = ef.L_sum(ff_opt_reg, work, algo, verbose=True)
    if algo in ['BO','SDM']:
        print('# grep1: FINAL L_sum: %10.8f'%Lval_sum)
        util.printff(ff_opt)
    elif algo == 'FDM':
        print('# FINAL L_sum(FD): %10.8f'%Lval_sum)
        Lval_sum = ef.L_sum(ff_opt_reg,  work, 'SDM', verbose=True)
        print('# grep1: FINAL L_sum(SD): %10.8f'%Lval_sum)
        util.printff(ff_opt)

    ## (5) output final state with algo=SD mode (even FD optimization mode)
    save_to_gjf_lmp(ff_opt,work,'opt')

    ## (6) draw pes
    if len(pes_dirs) >= 1:
        ffa_ = work['task_list'][0]['ffa']
        draw_pes(ff_ini,ff_opt,ffa_,work,pes_dirs)

    ## (7) final output
    return ff_opt

def noopt(ff_ini,work,algo):
    '''
    The function noopt() evaluates the force field after optimization by 
    calculating the L_sum.

    Args:
        ff_ini: a dictionary containing the initial force field parameters.
        work: a dictionary containing information about the optimization work.
        algo: a string specifying the optimization algorithm to be used.

    Returns:
        The function does not explicitly return any value, 
        but it prints the calculated value of the L_sum.
    '''
    root = work['root']
    idx  = work['idx']
    reg = work['reg']

    ## (1) calc init L_sum with algo=SD mode (even FD optimization mode)
    print('\n\n\n##### EVALUATION OF FF #####')
    ff_ini_reg = util.doreg(ff_ini,reg)
    Lval_sum = ef.L_sum(ff_ini_reg,  work, algo, verbose=True)
    print('# L_sum(%s): %10.8f'%(algo,Lval_sum))
    if algo == 'FDM':
       Lval_sum = ef.L_sum(ff_ini_reg,  work, 'SDM', verbose=True)
       print('# L_sum(SDM): %10.8f'%Lval_sum)

def save_to_gjf_lmp(ff_,work,label):
    ''' 
    The function saves the optimized structures with the force field
    in the following formats:
    (1) monomer
        Gaussian Input File (xxx.gjf)
        Lammps Input Files (xxx.data,xxx.in.settings)
    (2) crystal
        Gaussian Input File (xxx.gjf)
        Lammps Input Files (xxx.data,xxx.in.settings)
        XYX File for OVITO, VESTA and so on.  (xxx.xyz)
    
    Args:
        ff_: a dictionary containing the force field parameters.
        work: a dictionary containing information about the optimization work.
        label: a string specifying a label for the output files.

    Returns:
        The function does not explicitly return any value, but it saves the FF to the specified file formats.
    '''
    root = work['root']
    name = work['name']
    idx  = work['idx']
    task_list = work['task_list']
    taskids = work['taskids']
    task_monomer = task_list[0] # monomer
    sys_tgt_monomer = task_monomer['sys']
    ffa_ = task_monomer['ffa']
    atomtypelabels = task_monomer['atomtypelabels']
    for taskid,task in zip(taskids,task_list):
        if taskid==0 and task['type'] == 'opt': #monomer
            # output init states for monomer
            sys_opt = opt.opt_sys(ff_,sys_tgt_monomer,ffa_)
            gff.write_gjf(root + name + '_%sff%02d.gjf'%(label,idx),ff_,sys_opt,ffa_,atomtypelabels)
            lff.write_system_in_settings(root + name + '_%sff%02d.in.settings'%(label,idx),ff_,ffa_)
            lff.write_system_data(root + name + '_%sff%02d.data' \
                    %(label,idx),ff_,sys_opt,ffa_,atomtypelabels)
        elif taskid>=1 and task['type'] == 'opt_lat': # laternal
            sys_tgt = task['sys']
            ffa_lat = task['ffa']
            sys_opt = opt_lat.opt_lat_sys(ff_,sys_tgt,ffa_lat)
            gff.write_gjf(root + name + '_lat_%sff%02d.gjf'\
                    %(label,idx),ff_,sys_opt,ffa_lat,atomtypelabels)
            #gff.write_xyz(root + name + '_lat_%sff%02d.xyz'\
            #        %(label,idx),ff_,sys_opt,ffa_lat,atomtypelabels)
            lff.write_system_in_settings(root + name + '_lat_%sff%02d.in.settings'\
                    %(label,idx),ff_,ffa_lat)
            lff.write_system_data(root + name +  '_lat_%sff%02d.data'\
                    %(label,idx),ff_,sys_opt,ffa_lat,atomtypelabels)

def _draw_pes(ff_ini,ff_opt,ffa_,work,pes_dirs):
    '''
    The function draw_pes computes and plots the Potential Energy Surface (PES) of a molecule using two different force fields, ff_ini and ff_opt, and saves the results in various formats. It calculates energy components like bond, angle, dihedral, improper, coulomb, and van der Waals energies, and generates plots comparing the energies of the two force fields with Gaussian (g16) energies.
    
    Args:
    
    ff_ini (object): Initial force field.
    ff_opt (object): Optimized force field.
    ffa_ (object): Auxiliary force field object.
    work (dict): A dictionary containing the root directory and task-related information.
    pes_dirs (list): A list of directories containing the PES data.
    Returns:
    This function does not return any value but saves the following files to disk:
    
    PES plots comparing g16, initial and optimized force fields.
    PES component plots for the optimized force field.
    CSV files containing the computed PES data for initial and optimized force fields.
   '''



def draw_pes(ff_ini,ff_opt,ffa_,work,pes_dirs):
    root = work['root']
    idx = work['idx']
    task_list = work['task_list']
    task_monomer = task_list[0] # monomer
    sys_tgt_monomer = task_monomer['sys']
    natom = sys_tgt_monomer.coord.shape[1] # nmol x natom x 3dim
    nmolvec  = jnp.array([1])
    natomvec = jnp.array([natom])

    #GH = GaussianHandler()

    # initial
    dict_pes_ini=[]
    for ipes,pes_dirs in enumerate(pes_dirs):
        
        ipes_task=-1
        for task in task_list:
            if task['type'] == 'pes':ipes_task+=1
            if ipes_task == ipes: break

        sys_pes = task['sys_pes']
        energies = task['energies']
        print('# PES %s'%pes_dirs)
        files = os.listdir(pes_dirs)
        files.sort()

        nmod=1000
        ipoint = 0
        for file in files:
            if '.log' in file:
                ipoint+=1
        npoint = ipoint
        pesmat_ini=np.zeros((npoint,10),float)
        pesmat_opt=np.zeros((npoint,10),float)
        ipoint = -1
        for file in files:
            if '.log' in file:
                ipoint+=1
                g16energy=energies[ipoint]
                coord_ = jnp.reshape(sys_pes.coord[ipoint,:,:],(1,-1,3))
                sys_ = System(coord_)

                E_bond, E_angle,E_dihed, E_improper, E_coul, E_long, E_vdw = \
                    energy.energy_each_coord(ff_ini,sys_,ffa_)

                E_tot = energy.energy_coord(ff_ini,sys_,ffa_)

                pesmat_ini[ipoint,0] = np.floor(ipoint/nmod)
                pesmat_ini[ipoint,1] = ipoint%npoint
                pesmat_ini[ipoint,2] = g16energy
                pesmat_ini[ipoint,3] = E_tot
                pesmat_ini[ipoint,4] = E_bond
                pesmat_ini[ipoint,5] = E_angle
                pesmat_ini[ipoint,6] = E_dihed
                pesmat_ini[ipoint,7] = E_improper
                pesmat_ini[ipoint,8] = E_coul+E_long
                pesmat_ini[ipoint,9] = E_vdw


                E_bond, E_angle,E_dihed, E_improper, E_coul, E_long, E_vdw = \
                    energy.energy_each_coord(ff_opt,sys_,ffa_)

                E_tot = energy.energy_coord(ff_opt,sys_,ffa_)

                pesmat_opt[ipoint,0] = np.floor(ipoint/nmod)
                pesmat_opt[ipoint,1] = ipoint%npoint
                pesmat_opt[ipoint,2] = g16energy
                pesmat_opt[ipoint,3] = E_tot
                pesmat_opt[ipoint,4] = E_bond
                pesmat_opt[ipoint,5] = E_angle
                pesmat_opt[ipoint,6] = E_dihed
                pesmat_opt[ipoint,7] = E_improper
                pesmat_opt[ipoint,8] = E_coul+E_long
                pesmat_opt[ipoint,9] = E_vdw

        

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        imin = np.argmin(pesmat_ini[:,2])
        ax.set_ylim(-5,20)
        ax.plot(pesmat_ini[:,1], (pesmat_ini[:,2]-pesmat_ini[imin,2])*kcalM2Ha,\
                label='g16', linewidth=2)
        ax.plot(pesmat_ini[:,1], (pesmat_ini[:,3]-pesmat_ini[imin,3])*kcalM2Ha,\
                label='ini', linewidth=2)
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,3]-pesmat_opt[imin,3])*kcalM2Ha,\
                label='opt', linewidth=2)
        ax.set_ylabel('energy [kcal/Mol]')
        ax.set_xlabel('dihedral index')
        plt.legend()
        fig.savefig(root + 'pes%d_g16_ff%02d.png'%(ipes,idx))

        # each compontents of delff
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        imin = np.argmin(pesmat_ini[:,2])
        ax.set_ylim(-5,20)
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,3]-pesmat_opt[imin,3])*kcalM2Ha, label='tot')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,4]-pesmat_opt[imin,4])*kcalM2Ha, label='bond')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,5]-pesmat_opt[imin,5])*kcalM2Ha, label='angle')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,6]-pesmat_opt[imin,6])*kcalM2Ha, label='dihed')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,7]-pesmat_opt[imin,7])*kcalM2Ha, label='improper')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,8]-pesmat_opt[imin,8])*kcalM2Ha, label='coulomb')
        ax.plot(pesmat_opt[:,1], (pesmat_opt[:,9]-pesmat_opt[imin,9])*kcalM2Ha, label='vdw')
        ax.set_ylabel('energy [kcal/Mol]')
        ax.set_xlabel('dihedral index')
        plt.legend()
        fig.savefig(root + 'pes%d_compontents_ff%02d'%(ipes,idx))
        
        ## save to csv
        np.savetxt(root+'pes%d_ini%02d.csv'%(ipes,idx),pesmat_ini,delimiter=',',\
                header="period,point,g16,lmp,bond,angle,dihed,improper,coulomb,vdw")
        np.savetxt(root+'pes%d_opt%02d.csv'%(ipes,idx),pesmat_opt,delimiter=',',\
                header="period,point,g16,lmp,bond,angle,dihed,improper,coulomb,vdw")


def search_neighbors(sys_lat,ccutoff=10.0):
    ccutoff += 2.0
    coord   = sys_lat.coord
    lattice = sys_lat.lattice
    rall,latidx = alldists_lat(sys_lat)
    print('\n# cutoff for neighbors: %f'%ccutoff)
    print(rall.shape)

    nneighbors = jnp.sum(rall.ravel()<=ccutoff)

    nmol,natom,ncell,_,_=rall.shape
    
    neighbors = jnp.zeros((nmol,natom,ncell,nmol,natom,7),i16)
    #print('\n# neighbors.shape',neighbors.shape)

    def idx_neighbors(r,ccutoff,imol,iatom,latidx_each,jmol,jatom):
        flag_outer_cell = jnp.logical_or(latidx_each[0]!=0,latidx_each[1]!=0)
        flag_outer_cell = jnp.logical_or(flag_outer_cell,latidx_each[2]!=0)
        flag_UT_incell  = jnp.logical_or(imol<jmol,jnp.logical_and(imol==jmol,iatom<jatom))
        flag_UT         = jnp.logical_or(flag_outer_cell,flag_UT_incell)
        return jnp.where(jnp.logical_and(r<=ccutoff,flag_UT),\
                jnp.asarray([imol,iatom,latidx_each[0],latidx_each[1],latidx_each[2],jmol,jatom]),\
                jnp.asarray([-1,0,0,0,0,0,0]))

    vneighbor1 = vmap(idx_neighbors, (0,None,None,None,None,None,0), 0)
    vneighbor2 = vmap(vneighbor1   , (0,None,None,None,None,0,None), 0)
    vneighbor3 = vmap(vneighbor2   , (0,None,None,None,0,None,None), 0)
    vneighbor4 = vmap(vneighbor3   , (0,None,None,0,None,None,None), 0)
    vneighbor5 = vmap(vneighbor4   , (0,None,0,None,None,None,None), 0)

    neighbors = vneighbor5(rall,ccutoff,jnp.arange(nmol),jnp.arange(natom),latidx,\
            jnp.arange(nmol),jnp.arange(natom))

    neighbors = jnp.reshape(neighbors,(-1,7))
    neighbors = neighbors[neighbors[:,0]>=0,:]

    print('\n# num. of neighbors: %d'%neighbors.shape[0])
    return neighbors, latidx


@jit
def alldists_lat(sys_: System) -> Array:

    def dist(v0,v1,tv):
        dv = v0-(v1+tv)
        return jnp.sqrt(jnn.relu(jnp.dot(dv,dv)))

    idxs = jnp.array([-3,-2,-1,0,1,2,3])
    def vec(x,y,z): return jnp.array([x,y,z]) 
    vec1 = vmap(vec,  (0,None,None),0)
    vec2 = vmap(vec1, (None,0,None),0)
    vec3 = vmap(vec2, (None,None,0),0)
    latidx = vec3(idxs,idxs,idxs)
    latidx = jnp.reshape(latidx,(-1,3))

    tvs = latidx @ sys_.lattice

    vdist1 = vmap(dist  ,(None,0,None),0)
    vdist2 = vmap(vdist1,(None,0,None),0)
    vdistL = vmap(vdist2,(None,None,0),0)
    vdist3 = vmap(vdistL,(0,None,None),0)
    vdist4 = vmap(vdist3,(0,None,None),0)
    rall   = vdist4(sys_.coord,sys_.coord,tvs)

    #print(rall.shape)

    return rall,latidx



