from scipy import constants as const
from math import pi
import jax.numpy as jnp
from jax import vmap
from delff.objects import *


def read_system(in_settings_file,data_file,atomtypesets=None):
    """Reads system data from the settings and data files in the LAMMPS format. 

    Arguments:
        in_settings_file (str): Path to the settings file.
        data_file (str): Path to the data file.
        atomtypesets (list, optional): List of atom type sets. If None, it will be created in the function. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - ff_ (ForceField): Updated ForceField object.
            - sys_ (System): System object containing the molecular system data.
            - ffa_ (ForceFieldAssignments): Updated ForceFieldAssignments object.
            - atomtypelabels (list): List of atom type labels.
    """
    nmolvec=jnp.asarray([1])
    natom = len([v for eachset in atomtypesets for v in eachset ])
    natomvec=jnp.asarray([natom])

    _, dihedralmasks,dihedralphis = read_system_in_settings(in_settings_file,charges=None)

    # atomtypelabels is a str-list which is NOT compatible with JAX.
    ffa_, sys_, charges, atomtypelabels = read_system_data(data_file,dihedralmasks,dihedralphis,nmolvec,natomvec)
    ff_,_,_ = read_system_in_settings(in_settings_file,charges=charges)


    # create atomtypesets if necessary
    if atomtypesets is None:
        natomtypes = jnp.max(ffa_.atomtypes)+1
        atomtypesets = [[] for i in range(natomtypes)]
        for iatom,itype in enumerate(ffa_.atomtypes):
            atomtypesets[itype].append(iatom)

    # charge and pairs averaging based on atomtypesets
    natommol = len(ffa_.atomtypes)
    pairsnb   = make_nbpairs(ff_.pairs,ffa_.atomtypes,atomtypesets)
    chargesnb = make_nbcharges(ff_.charges,ffa_.atomtypes,atomtypesets)
    massesnb  = make_nbmasses(ffa_.masses,ffa_.atomtypes,atomtypesets)
    ff_ = update(ff_,charges=chargesnb)
    ff_ = update(ff_,pairs=pairsnb)
    ffa_ = update(ffa_,masses=massesnb)

    atomtypelabels = make_atomtypelabels(atomtypelabels,ffa_.atomtypes,atomtypesets)

    # update atomtypes and labels
    atomtypes = jnp.zeros(natommol,i32)
    for itype, atomset in enumerate(atomtypesets):
        for iatom in atomset:
            atomtypes = atomtypes.at[iatom].set(itype)
    ffa_ = update(ffa_,atomtypes=atomtypes)

    return ff_, sys_, ffa_, atomtypelabels


def read_system_in_settings(in_settings_file,charges=None):
    """Reads a system settings file in the LAMMPS format.

    Arguments:
        in_settings_file (str): Path to the settings file.
        charges (jax.numpy.ndarray, optional): Array containing charges. If None, no charges will be assigned. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - ff_ (ForceField): ForceField object containing the force field parameters.
            - dihedralmasks (jax.numpy.ndarray): Array of dihedral masks.
            - dihedralphis (jax.numpy.ndarray): Array of dihedral angles.
    """

    nbondtypes=0
    nangletypes=0
    ndihedraltypes=0
    npairtypes=0
    nimpropertypes=0
    with open(in_settings_file) as f:
      for line in f:
        words = line.rstrip().split()  
        if len(words)>=1:
          if words[0] == 'bond_coeff': nbondtypes+=1
          elif words[0] == 'angle_coeff': nangletypes+=1
          elif words[0] == 'dihedral_coeff': ndihedraltypes+=1
          elif words[0] == 'improper_coeff': nimpropertypes+=1
          elif words[0] == 'pair_coeff': npairtypes+=1

    bondtypes = jnp.zeros((nbondtypes,2),f64)
    angletypes = jnp.zeros((nangletypes,2),f64)
    dihedralks = jnp.zeros((ndihedraltypes,4),f64)
    dihedralphis = jnp.zeros((ndihedraltypes,4),f64)
    dihedralmasks = jnp.ones((ndihedraltypes,4),i32) # utilize all 4 dihedral factors
    impropertypes = jnp.zeros((nimpropertypes,3),f64)
    pairs = jnp.zeros((npairtypes,2),f64)

    with open(in_settings_file) as f:
      for line in f:
        words = line.rstrip().split()  
        if words[0] == 'bond_coeff':
            idx = i32(words[1])
            k   = f64(words[3])
            req = f64(words[4])
            bondtypes = bondtypes.at[idx-1,0].set(k)
            bondtypes = bondtypes.at[idx-1,1].set(req)

        if words[0] == 'angle_coeff':
            idx = i32(words[1])
            k   = f64(words[3])
            thetaeq = f64(words[4])
            angletypes = angletypes.at[idx-1,0].set(k)
            angletypes = angletypes.at[idx-1,1].set(thetaeq)

        if words[0] == 'dihedral_coeff':
            idx = i32(words[1])
            v1=0.
            v2=0.
            v3=0.
            v4=0.
            phieq1=0.
            phieq2=0.
            phieq3=0.
            phieq4=0.
            for i in range(i32(words[3])):
                if i32(words[3*i+5]) == 1:
                    v1 = f64(words[3*i+4])
                    phieq1 = f64(words[3*i+6])
                elif i32(words[3*i+5]) == 2:
                    v2 = f64(words[3*i+4])
                    phieq2 = f64(words[3*i+6])
                elif i32(words[3*i+5]) == 3:
                    v3 = f64(words[3*i+4])
                    phieq3 = f64(words[3*i+6])
                elif i32(words[3*i+5]) == 4:
                    v4 = f64(words[3*i+4])
                    phieq4 = f64(words[3*i+6])
                else:
                    raise ValueError('N in dihedral > 5 is not compatible.')
            dihedralks = dihedralks.at[idx-1,0].set(v1)
            dihedralphis = dihedralphis.at[idx-1,0].set(phieq1)
            dihedralks = dihedralks.at[idx-1,1].set(v2)
            dihedralphis = dihedralphis.at[idx-1,1].set(phieq2)
            dihedralks = dihedralks.at[idx-1,2].set(v3)
            dihedralphis = dihedralphis.at[idx-1,2].set(phieq3)
            dihedralks = dihedralks.at[idx-1,3].set(v4)
            dihedralphis = dihedralphis.at[idx-1,3].set(phieq4)

        if words[0] == 'pair_coeff': # pair = Lenard Jones potential
            atomtype0 = i32(words[1])
            atomtype1 = i32(words[2])
            assert atomtype0 == atomtype1
            epsilon = f64(words[4])
            sigma = f64(words[5])
            pairs = pairs.at[atomtype0-1,0].set(epsilon)
            pairs = pairs.at[atomtype0-1,1].set(sigma)
            
        if words[0] == 'improper_coeff':
            idx = i32(words[1])

            # convert k,d, and n in lammps to mag, po, and period in gaussian
            k = f64(words[3])
            mag = k
            d = i32(words[4])
            n = i32(words[5])
            if d == -1: po = 180.0
            elif d == 1: po = 0.0
            period = n
            impropertypes = impropertypes.at[idx-1,0].set(mag)
            impropertypes = impropertypes.at[idx-1,1].set(po)
            impropertypes = impropertypes.at[idx-1,2].set(period)


    #print('\n# dihedralphis in lammpsff.py: ',dihedralphis)
    #print('\n# dihedralks in lammpsff.py: ',dihedralks)
    ff_  = ForceField(bondtypes,angletypes,dihedralks,impropertypes,pairs,charges)
    return ff_,dihedralmasks,dihedralphis

def make_nbpairs(pairs,atomtypes,atomtypesets):
    """Generates non-bonded pairs based on atom type sets.

    Arguments:
        pairs (jax.numpy.ndarray): Array of pair coefficients.
        atomtypes (jax.numpy.ndarray): Array of atom types.
        atomtypesets (list): List of atom type sets.

    Returns:
        jax.numpy.ndarray: Array of non-bonded pairs.
    """
    nnbat   = len(atomtypesets)
    pairsnb = jnp.zeros((nnbat,2),f64)
    for inbat in range(nnbat):
        atomset = atomtypesets[inbat]
        pair0 = pairs[atomtypes[atomset[0]],0]
        pair1 = pairs[atomtypes[atomset[0]],1]
        pairsnb = pairsnb.at[inbat,0].set(pair0)
        pairsnb = pairsnb.at[inbat,1].set(pair1)
    return pairsnb

def make_nbmasses(masses,atomtypes,atomtypesets):
    """Generates masses based on atom type sets. 

    Arguments:
        masses (jax.numpy.ndarray): Array of atom masses.
        atomtypes (jax.numpy.ndarray): Array of atom types.
        atomtypesets (list): List of atom type sets.

    Returns:
        jax.numpy.ndarray: Array of non-bonded masses.
    """
    nnbat   = len(atomtypesets)
    massesnb = jnp.zeros((nnbat),f64)
    for inbat in range(nnbat):
        atomset = atomtypesets[inbat]
        massenb_ = masses[atomtypes[atomset[0]]]
        massesnb = massesnb.at[inbat].set(massenb_)
    return massesnb

    
def make_nbcharges(charges,atomtypes,atomtypesets):
    """Generates charges based on atom type sets. 

    Arguments:
        charges (jax.numpy.ndarray): Array of charges.
        atomtypes (jax.numpy.ndarray): Array of atom types.
        atomtypesets (list): List of atom type sets.

    Returns:
        jax.numpy.ndarray: Array of non-bonded charges.
    """
    nnbat   = len(atomtypesets)
    chargesnb = jnp.zeros((nnbat),f64)
    for inbat in range(nnbat):
        atomset = jnp.array(atomtypesets[inbat])
        chargeset = charges[atomset]
        avecharge = jnp.mean(chargeset)
        chargesnb = chargesnb.at[inbat].set(avecharge)
    return chargesnb
    
def make_atomtypelabels(atomtypelabels,atomtypes,atomtypesets):
    """Generates atom type labels based on atom type sets. 

    Arguments:
        atomtypelabels (list): List of initial atom type labels.
        atomtypes (jax.numpy.ndarray): Array of atom types.
        atomtypesets (list): List of atom type sets.

    Returns:
        list: Updated list of atom type labels.
    """
    atomtypelabels_=[]
    nnbat   = len(atomtypesets)
    for inbat in range(nnbat):
        isUpdate=False
        atomset = atomtypesets[inbat]
        atomtype = atomtypes[atomset[0]]
        atomtypelabel = atomtypelabels[atomtype]
        #print(atomtype,atomtypelabel)
        for i in range(100):
            if i >= 1: isUpdate=True
            cand_atomtypelabel = atomtypelabel + '%d'%i
            if cand_atomtypelabel in atomtypelabels_:
                pass
            else:
                atomtypelabels_.append(cand_atomtypelabel)
                break;
            assert i < 99

    atomtypelabels__ = atomtypelabels_
    for ilabel0, atomtypelabel0 in enumerate(atomtypelabels_):
        flag_dupl = False
        for ilabel1, atomtypelabel1 in enumerate(atomtypelabels_):
            if ilabel0 != ilabel1:
                if atomtypelabel0[:-1] == atomtypelabel1[:-1]: flag_dupl = True

        #print(flag_dupl)
        if not(flag_dupl):
            atomtypelabels__[ilabel0]=atomtypelabel0[:-1]

    print(atomtypelabels__)

    return atomtypelabels__



def read_system_data(data_file,dihedralmasks,dihedralphis,nmolvec,natomvec):
    """Reads system data from the given file.

    Arguments:
        data_file (str): Path to the data file.
        dihedralmasks (jax.numpy.ndarray): Array of dihedral masks.
        dihedralphis (jax.numpy.ndarray): Array of dihedral angles.
        nmolvec (jax due to similar content.
        Returns:
            tuple: A tuple containing ff_ (ForceField), sys_ (System), ffa_ (ForceFieldAssignments), and atomtypelabels (List of str). These objects contain the information for the system, the force field, the force field assignments for each atom, and the atom type labels, respectively.
    """

    nbonds,nangles,ndihedrals,nimpropers=0,0,0,0
    lBonds,lAngles,lDihedrals,lImpropers=0,0,0,0

    with open(data_file) as f:
      for iline,line in enumerate(f):
          words = line.rstrip().split()
          if len(words)==1:
              if words[0] == 'Masses': lMasses = iline
              elif words[0] == 'Atoms': lAtoms = iline
              elif words[0] == 'Bonds': lBonds = iline
              elif words[0] == 'Angles': lAngles = iline
              elif words[0] == 'Dihedrals': lDihedrals = iline
              elif words[0] == 'Impropers': lImpropers = iline
          elif len(words)==2:
              if words[1]== 'atoms': natoms=i32(words[0])
              elif words[1]== 'bonds': nbonds=i32(words[0])
              elif words[1]== 'angles': nangles=i32(words[0])
              elif words[1]== 'dihedrals': ndihedrals=i32(words[0])
              elif words[1]=='impropers': nimpropers=i32(words[0])
          elif len(words)==3:
              if words[1] == 'atom' and words[2] == 'types': natomtypes = i32(words[0])
          elif len(words) >=4:
              if 'xy' in words: raise ValueError('The lattice must be an orthorhombic or cubic cell.')
             
    atomtypes = jnp.zeros((natoms),i32)
    masses    = jnp.zeros((natomtypes),f64)
    atomtypelabels = []
    charges   = jnp.zeros((natoms),f64)
    #charges_eachatom = jnp.zeros((natoms),f64)
    #charges   = jnp.zeros((natomtypes),f64)
    coord_    = jnp.zeros((natoms,3),f64)
    bonds     = jnp.zeros((nbonds,3),i32)
    angles    = jnp.zeros((nangles,4),i32)
    dihedrals = jnp.zeros((ndihedrals,5),i32)
    impropers = jnp.zeros((nimpropers,5),i32)

    with open(data_file) as f:
      for iline,line in enumerate(f):
          words = line.rstrip().split()

          # Masses
          if lMasses + 2 <= iline and iline <= lMasses + natomtypes + 1:
              masses=masses.at[i32(words[0])-1].set(f64(words[1]))
              atomtypelabels.append(words[3].upper())

          # Atoms
          if lAtoms + 2 <= iline and iline <= lAtoms + natoms + 1:
              if i32(words[1]) > 1: ValueError('Thid system.data file can only include a single molecule.')
              atomtypes=atomtypes.at[i32(words[0])-1].set(i32(words[2])-1)
              #charges_eachatom=charges_eachatom.at[i32(words[0])-1].set(f64(words[3]))
              charges=charges.at[i32(words[0])-1].set(f64(words[3]))
              coord_=coord_.at[i32(words[0])-1,0].set(f64(words[4]))
              coord_=coord_.at[i32(words[0])-1,1].set(f64(words[5]))
              coord_=coord_.at[i32(words[0])-1,2].set(f64(words[6]))
             
          # Bonds
          if lBonds + 2 <= iline and iline <= lBonds + nbonds + 1:
              bonds=bonds.at[i32(words[0])-1,0].set(i32(words[1])-1)
              bonds=bonds.at[i32(words[0])-1,1].set(i32(words[2])-1)
              bonds=bonds.at[i32(words[0])-1,2].set(i32(words[3])-1)
             
          # Angles
          if lAngles + 2 <= iline and iline <= lAngles + nangles + 1:
              angles=angles.at[i32(words[0])-1,0].set(i32(words[1])-1)
              angles=angles.at[i32(words[0])-1,1].set(i32(words[2])-1)
              angles=angles.at[i32(words[0])-1,2].set(i32(words[3])-1)
              angles=angles.at[i32(words[0])-1,3].set(i32(words[4])-1)
          
          # Dihedrals
          if lDihedrals + 2 <= iline and iline <= lDihedrals + ndihedrals + 1:
              dihedrals=dihedrals.at[i32(words[0])-1,0].set(i32(words[1])-1)
              dihedrals=dihedrals.at[i32(words[0])-1,1].set(i32(words[2])-1)
              dihedrals=dihedrals.at[i32(words[0])-1,2].set(i32(words[3])-1)
              dihedrals=dihedrals.at[i32(words[0])-1,3].set(i32(words[4])-1)
              dihedrals=dihedrals.at[i32(words[0])-1,4].set(i32(words[5])-1)

          # Impropers 
          # convert the tyle in https://docs.lammps.org/improper_cvff.html
          # to ImpTrs in https://gaussian.com/mm/
          if lImpropers + 2 <= iline and iline <= lImpropers + nimpropers + 1:
              impropers=impropers.at[i32(words[0])-1,0].set(i32(words[1])-1)
              impropers=impropers.at[i32(words[0])-1,1].set(i32(words[2])-1)
              impropers=impropers.at[i32(words[0])-1,2].set(i32(words[3])-1)
              impropers=impropers.at[i32(words[0])-1,3].set(i32(words[4])-1)
              impropers=impropers.at[i32(words[0])-1,4].set(i32(words[5])-1)

    #for iatom, charge in enumerate(charges_eachatom):
    #    charges = charges.at[atomtypes[iatom]].set(charges_eachatom[iatom])

    #print('charges',charges)

    natoms = atomtypes.shape[0]
    adjmat012, adjmat3 = generate_scale_mat(bonds,natoms)

    coord_ = jnp.reshape(coord_,(nmolvec[0],-1,3))
    sys_  = System(coord_)
    ffa_  = ForceFieldAssignments(atomtypes,masses,bonds,angles,dihedrals,dihedralmasks,dihedralphis,impropers,adjmat012,adjmat3,nmolvec,natomvec)

    #print(ffa_.atomtypes)
    #print(atomtypes)
    return ffa_, sys_, charges, atomtypelabels

def generate_scale_mat(bonds, natoms):
    """Generates matrices that represent scaled bonding connections between atoms in a molecular system.

    Arguments:
        bonds (ndarray): Array of bonds where each bond is represented by a pair of atom indices.
        natoms (int): Total number of atoms in the system.

    Returns:
        tuple: A tuple of matrices that represent scaled bonding connections. 
               The first matrix contains bonding connections of distance 1 and 2. 
               The second matrix contains bonding connections of distance 3.
    """

    bonds = bonds
    nbonds = bonds.shape[0]

    # adjmat01
    adjmat01 = jnp.eye(natoms,dtype=i32)
    adjmat012 = jnp.eye(natoms,dtype=i32)
    adjmat3 = jnp.eye(natoms,dtype=i32)

    for ibond in range(nbonds): 
        adjmat01=adjmat01.at[bonds[ibond,1],bonds[ibond,2]].set(1)
        adjmat01=adjmat01.at[bonds[ibond,2],bonds[ibond,1]].set(1)

    for iatom in range(natoms):
        dist1vec = adjmat01[iatom,:]
        adjmat012=adjmat012.at[iatom,:].set(dist1vec) # dist=1
        for jatom in range(natoms):
            if dist1vec[jatom] == 1: # connect directly
                adjmat012 = adjmat012.at[iatom,:].set(adjmat012[iatom,:]+adjmat01[jatom,:]) # dist=2
    
    adjmat012 = vmap(vmap(lambda x: i32(x>=1),0,0),1,1)(adjmat012)

    for iatom in range(natoms):
        dist2vec = adjmat012[iatom,:]
        adjmat3=adjmat3.at[iatom,:].set(dist2vec) # dist=2
        for jatom in range(natoms):
            if dist2vec[jatom] >= 1: #  connect within <2 topological distances
                adjmat3 = adjmat3.at[iatom,:].set(adjmat3[iatom,:]+adjmat01[jatom,:]) # dist=3

    adjmat3 = vmap(vmap(lambda x: i32(x>=1),0,0),1,1)(adjmat3)
    adjmat3 -= adjmat012

    return adjmat012, adjmat3
          

          



def write_system_in_settings(in_settings_file,ff_,ffa_,charges=None):
    """Writes the system settings, including force field parameters and coefficients, into a LAMMPS in_settings file.

    Arguments:
        in_settings_file (str): Name of the output settings file.
        ff_ (ForceField): ForceField object containing the parameters for the force field.
        ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
        charges (ndarray, optional): If given, includes charge information. Defaults to None.
    """

    nbondtypes     = ff_.bondtypes.shape[0]
    nangletypes    = ff_.angletypes.shape[0]
    ndihedraltypes = ff_.dihedralks.shape[0]
    nimpropertypes = ff_.impropertypes.shape[0]
    npairs         = ff_.pairs.shape[0]

    with open(in_settings_file,'w') as f:
        for i in range(npairs):
            #f.write('pair_coeff %d %d lj/charmm/coul/long %16.12f %16.12f\n'%(i+1,i+1,ff_.pairs[i,0],ff_.pairs[i,1]))
            f.write('pair_coeff %d %d lj/charmm/coul/long %16.12f %16.12f\n'%(i+1,i+1,ff_.pairs[i,0],ff_.pairs[i,1]))

        for i in range(nbondtypes):
            f.write('bond_coeff %d harmonic %9.5f %9.5f\n'%(i+1,ff_.bondtypes[i,0],ff_.bondtypes[i,1]))

        for i in range(nangletypes):
            f.write('angle_coeff %d harmonic %9.5f %9.5f\n'%(i+1,ff_.angletypes[i,0],ff_.angletypes[i,1]))

        for i in range(ndihedraltypes):
            nterms = jnp.sum(ffa_.dihedralmasks[i,:])
            f.write('dihedral_coeff %d fourier %d '%(i+1,nterms))
            for j in range(4):
                if ffa_.dihedralmasks[i,j]==1:
                    f.write('%9.5f %d %9.5f'%(ff_.dihedralks[i,j],j+1,ffa_.dihedralphis[i,j]))
            f.write('\n')

        for i in range(nimpropertypes):
            if ff_.impropertypes[i,1] > 179.9: d = -1
            elif ff_.impropertypes[i,1] < 0.1: d = 1 
            else: raise ValueError
            f.write('improper_coeff %d cvff %9.5f %d %d\n' \
                    %(i+1,ff_.impropertypes[i,0],d,ff_.impropertypes[i,2]))

def write_system_data(data_file,ff_,sys_,ffa_,atomtypelabels):
    """Writes the system data including atom types, bond types, coordinates, masses and force field assignments into a LAMMPS data file.

    Arguments:
        data_file (str): Name of the output data file.
        ff_ (ForceField): ForceField object containing the parameters for the force field.
        sys_ (System): System object containing the information for the system.
        ffa_ (ForceFieldAssignments): ForceFieldAssignments object containing the force field assignments for each atom in the system.
        atomtypelabels (List of str): List of atom type labels.
    """

    with open(data_file,'w') as f:
        f.write('LAMMPS Description\n\n')

        # block1 nbonds, nangles, ...
        nmol = sys_.coord.shape[0]
        natoms = ffa_.natomvec[0]
        nbonds = ffa_.bonds.shape[0]
        nangles = ffa_.angles.shape[0]
        ndihedrals = ffa_.dihedrals.shape[0]
        nimpropers = ffa_.impropers.shape[0]
        f.write('%d atoms\n'%(natoms*nmol))
        f.write('%d bonds\n'%(nbonds*nmol))
        f.write('%d angles\n'%(nangles*nmol))
        if ndihedrals>0: f.write('%d dihedrals\n'%(ndihedrals*nmol))
        if nimpropers>0: f.write('%d impropers\n'%(nimpropers*nmol))
        f.write('\n')

        # block2 nbondtypes, nangletypes,...
        natomtypes = len(atomtypelabels)
        nbondtypes     = ff_.bondtypes.shape[0]
        nangletypes    = ff_.angletypes.shape[0]
        ndihedraltypes = ff_.dihedralks.shape[0]
        nimpropertypes = ff_.impropertypes.shape[0]
        npairs         = ff_.pairs.shape[0]
        f.write('%d atom types\n'%natomtypes)
        f.write('%d bond types\n'%nbondtypes)
        f.write('%d angle types\n'%nangletypes)
        if ndihedrals>0: f.write('%d dihedral types\n'%ndihedraltypes)
        if nimpropers>0: f.write('%d improper types\n'%nimpropertypes)
        f.write('\n')

        # lattice
        boxsize=100
        halfboxsize=boxsize/2
        f.write('%9.5f %9.5f xlo xhi\n'%(-halfboxsize,halfboxsize))
        f.write('%9.5f %9.5f ylo yhi\n'%(-halfboxsize,halfboxsize))
        f.write('%9.5f %9.5f zlo zhi\n'%(-halfboxsize,halfboxsize))
        f.write('\n')

        # masses
        f.write('Masses\n\n')
        assert len(atomtypelabels) == len(ffa_.masses)
        for iatomtype, atomtypelabel in enumerate(atomtypelabels):
            f.write("%d %9.5f # %s\n"%(iatomtype+1,ffa_.masses[iatomtype],atomtypelabel))
        f.write('\n')
        
        # atoms
        f.write('Atoms\n\n')
        #assert sys_.coord.shape[0] == 1 # This fucntion is now limited to a single molecule.
        assert sys_.coord.shape[1] == len(ffa_.atomtypes)
        assert jnp.max(ffa_.atomtypes)+1 == len(ff_.charges)
        natom = len(ffa_.atomtypes)
        #print(sys_.coord)
        for imol in range(nmol):
            for iatom in range(natom):
                f.write("%d %d %d %9.5f %9.5f %9.5f %9.5f\n"%(imol*natom+iatom+1,imol+1,ffa_.atomtypes[iatom]+1,ff_.charges[ffa_.atomtypes[iatom]],\
                    sys_.coord[imol,iatom,0],sys_.coord[imol,iatom,1],sys_.coord[imol,iatom,2]))
        f.write('\n')

        # bond
        f.write('Bonds\n\n')
        assert nbonds == ffa_.bonds.shape[0]
        for imol in range(nmol):
          for i in range(nbonds):
            f.write("%d %d %d %d\n"%(i+1,ffa_.bonds[i,0]+1,ffa_.bonds[i,1]+1+imol*natom,ffa_.bonds[i,2]+1+imol*natom))
        f.write('\n')

        # angle
        f.write('Angles\n\n')
        assert nangles == ffa_.angles.shape[0]
        for imol in range(nmol):
          for i in range(nangles):
            f.write("%d %d %d %d %d\n"%(i+1,ffa_.angles[i,0]+1,ffa_.angles[i,1]+1+imol*natom,ffa_.angles[i,2]+1+imol*natom,\
                    ffa_.angles[i,3]+1+imol*natom))
        f.write('\n')

        # dihedral
        if ndihedrals>0:
            f.write('Dihedrals\n\n')
            assert ndihedrals == ffa_.dihedrals.shape[0]
            for imol in range(nmol):
              for i in range(ndihedrals):
                f.write("%d %d %d %d %d %d\n"%(i+1,ffa_.dihedrals[i,0]+1,ffa_.dihedrals[i,1]+1+imol*natom,ffa_.dihedrals[i,2]+1+imol*natom,\
                        ffa_.dihedrals[i,3]+1+imol*natom,ffa_.dihedrals[i,4]+1+imol*natom))
            f.write('\n')

        # improper
        if nimpropers>0:
            f.write('Impropers\n\n')
            assert nimpropers == ffa_.impropers.shape[0]
            for imol in range(nmol):
              for i in range(nimpropers):
                f.write("%d %d %d %d %d %d\n"%(i+1,ffa_.impropers[i,0]+1,ffa_.impropers[i,1]+1+imol*natom,ffa_.impropers[i,2]+1+imol*natom,\
                        ffa_.impropers[i,3]+1+imol*natom,ffa_.impropers[i,4]+1+imol*natom))
            f.write('\n')

