import os,sys,shutil
import numpy as np
sys.path.append("/work/user/nakano/osc/mos/")
from mos.ext.gaussian import GaussianZmatHandler, GaussianCartHandler
from mos.ext.lammps import LammpsFF,LammpsHandler
from mos.ext.xyz import XYZHandler
from mos.app import Monomer, MixMols
from mos.infra import ZmatOperator, System, MolType, Mol, Periodic
import time


# (1) Single Point calc of  the monomer with g16
def calcMonomerG16_cart(molname):
    gjffile = molname+'.gjf'
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    GCH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='optd2') 
    # wB97X-D/6-311++G(d,p) = Horton2019_JCIM_QUBEKit D=long distance
    #GCH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='opt')

def calcMonomerG16_zmat(molname):
    gjffile = molname+'.gjf'
    GZH=GaussianZmatHandler()
    system0=GZH.getSystem(gjffile)
    GZH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='optd2')
    # wB97X-D/6-311++G(d,p) = Horton2019_JCIM_QUBEKit D=long distance
    #GZH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='opt')

# (2) Calc charge with g16
def calcChargeG16(molname):
    gjffile = molname+'.gjf'
    logfile=gjffile.rstrip('gjf')+'log'
    shutil.copyfile(gjffile,'chg_'+gjffile)
    shutil.copyfile(logfile,'chg_'+logfile)
    #GCH=GaussianCartHandler()
    #system1=GCH.getSystemLog('chg_'+logfile)
    #GCH.run(system1,calcname='chg_'+logfile.rstrip('log').rstrip('.'),calctype='chgd2nof')
    GZH=GaussianZmatHandler()
    system1=GZH.getSystemLog('chg_'+logfile)
    GZH.run(system1,calcname='chg_'+logfile.rstrip('log').rstrip('.'),calctype='chgd2nof')
    #GZH.run(system1,calcname='chg_'+logfile.rstrip('log').rstrip('.'),calctype='chg')

# (3) write out ESP charge
def writeoutESPCharge(chg_logfile):
    GCH=GaussianCartHandler()
    GCH.writeout_ESP_charge(chg_logfile)

# (4) make initial force field
def make_init_ff(molname,logfile,xyzfile):
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/mol22lt.pl","./mol22lt.pl")
    LFF=LammpsFF()
    LFF.setup_init_FF(logfile,xyzfile)
    shutil.move('system.data','%s.data'%molname)
    shutil.move('system.in.settings','%s.in.settings'%molname)
    shutil.move('system.in.init','%s.in.init'%molname)

# (5)
def clean_ff(system_data_file,system_settings_file):
    LFF=LammpsFF()
    LFF.read_system_data(system_data_file)
    LFF.write_system_data(system_data_file)
    LFF.read_system_settings(system_settings_file)
    LFF.write_system_settings(system_settings_file)

#(6) getSystem from system_data_file    
def test_lammps_run(molname,ffname):
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)

    system_data = molname + '.data'
    system_in_settings = molname + '.in.settings'
    lattice = 100.0*np.eye(3)
    system0.setLattice(lattice)
    system0.getMolType(0).molname=molname
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/sp.in","./sp.in")
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/opt.in","./opt.in")
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/system.in.init","./system.in.init")
    #shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/coulcharmm.in.init","./system.in.init")
    #molname=system0.getMolType(0).molname
    LH=LammpsHandler()
    LH.run(system0,ffname=ffname,infile='opt.in',calcname=molname+'_opt')
    # Improperが力場で設定されていない場合はsysytem.in.initからimproperの行を削除しないとエラーになる。

def g16cartopt(gjffile):
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    GCH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='optd2')
    
def convert_crystal_gjf2xyz(gjffile):
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    system0=Periodic.connectMolAcrossBoundary(system0)
    lattice=system0.getLattice()
    system0.setLattice(lattice)
    xh0=XYZHandler()
    xh0.write(system0,'%s.xyz'%molname)

def opt_crystal_lmp(gjffile,ffname,nmollist=[4],natomlist=[28]):
    LH=LammpsHandler()
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    system0=Periodic.connectMolAcrossBoundary(system0)
    #shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/opt.in","./opt.in")
    #shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/system.in.init","./system.in.init")
    # ffnam.data must A SINGLE molecule data file 
    LH.run(system0,ffname=ffname,infile='optwcell.in',calcname=molname+'',test=False)
    time.sleep(20)
    system1=LH.getSystemLog('optwcell.trj',nmollist=nmollist,natomlist=natomlist)
    #print(system1.getLattice())
    system1=Periodic.connectMolAcrossBoundary(system1)
    GCH.run(system1,calcname='optwcell_'+gjffile.rstrip('gjf').rstrip('.'),calctype='sp',test=True)

def supercell(molname,supercell=[3,3,3],vacant=[0,0,0],shift=[0,0,0],ffname=""):
    gjffile=molname + '.gjf'
    #LH=LammpsHandler()
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    system0=Periodic.supercell(system0,supercell,vacant=vacant,shift=shift)
    #system0=Periodic.connectMolAcrossBoundary(system0)
    supercell_label = str(supercell[0])+str(supercell[1])+str(supercell[2])
    GCH.run(system0,calcname=molname+supercell_label,calctype='optd2',test=True)

    xh0=XYZHandler()
    xh0.write(system0,'%s.xyz'%(molname+supercell_label))

    if len(ffname)>=0:
        LH=LammpsHandler()
        LH.run(system0,ffname=ffname,infile='sp.in',calcname=molname+'_sp',test=True)


# (11)
def npt(molname,ffname,supercell=[1,1,1]):
    gjffile=molname + '.gjf'
    LH=LammpsHandler()
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    system0=Periodic.supercell(system0,supercell)
    system0=Periodic.connectMolAcrossBoundary(system0)
    #system0=LH.getSystemLog('opt_222.trj',nmollist=[16],natomlist=[24])
    #system0=Periodic.supercell(system0,[6,6,3]) ############### supercell must follow connectMolAcrossBoundary
    #lattice=system0.getLattice()
    #lattice[0,:] *= 3.0 
    #system0.setLattice(50.0*np.eye(3))
    #xh0=XYZHandler()
    #xh0.write(system0,molname + '_cluster.xyz')
    #print(system0.getLattice())
    #GCH.run(system0,calcname='optrelax_443.gjf'.rstrip('gjf').rstrip('.'),calctype='sp',test=True)
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/system.in.init","./system.in.init")
    shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/lammps/npt300K.in","./npt300K.in")
    LH.run(system0,ffname=ffname,infile='label_npt300K.in',calcname=molname+'_npt300K',test=False)
    #LH.run(system0,ffname=ffname,infile='opt.in',calcname=molname+'_opt',test=False)

def trj2gjf(molname,trjname,nmollist=[4],natomlist=[28]):
    LH=LammpsHandler()
    GCH=GaussianCartHandler()
    system1=LH.getSystemLog(trjname,nmollist=nmollist,natomlist=natomlist)
    #system1=Periodic.connectMolAcrossBoundary(system1)
    #system1.setLattice(None)
    GCH.run(system1,calcname=molname+'_trj',calctype='optd2nof',test=True)
    xh0=XYZHandler()
    xh0.write(system1,molname + '_trj.xyz')

def calcDimerG16_BSSE(gjfname):
    gjffile = molname+'.gjf'
    GCH=GaussianCartHandler()
    system0=GCH.getSystem(gjffile)
    GCH.run(system0,calcname=gjffile.rstrip('gjf').rstrip('.'),calctype='optd2BSSE')

def change_trj_for_ovito(trjfile):
    fi = open(trjfile)
    fo = open('ovito_' + trjfile,'w')
    for line in fi:
        if 'ITEM: ATOMS id mol type q x y z ix iy iz element' in line:
            fo.write('ITEM: ATOMS id mol type_ q x y z ix iy iz type\n')
        else:
            fo.write(line)
    fi.close()
    fo.close()



if __name__ == '__main__':

    jobtype=int(sys.argv[1])
    if jobtype in [14]: pass
    else:
        if sys.argv[2][-4:] == '.gjf':
            gjfname=sys.argv[2]
            molname = gjfname.rstrip('gjf').rstrip('.')
        else:
            molname=sys.argv[2]
            gjffile=molname + '.gjf'
    #gjffile=sys.argv[1]

    gjffile=molname + '.gjf'
    #molname = gjffile.rstrip('gjf').rstrip('.')

    if jobtype ==0:
        calcMonomerG16_cart(molname)

    elif jobtype ==1:
        calcMonomerG16_zmat(molname)

    elif jobtype == 2:
        calcChargeG16(molname)

    elif jobtype == 3:
        writeoutESPCharge('chg_'+molname + '.log')

    elif jobtype == 4:
        make_init_ff(molname,sys.argv[3],sys.argv[4])

    elif jobtype == 5:
        clean_ff('%s.data'%molname,'%s.in.settings'%molname)
  
    elif jobtype == 6:
        # ffnameじで指定するffname.dataは単分子のデータでなければならない。
        ffname = sys.argv[3]
        #nmol = int(sys.argv[4])
        #natom = int(sys.argv[5])
        test_lammps_run(molname,ffname)
    elif jobtype == 7:
        if len(sys.argv) >=5:
            check = (sys.argv[4] == 'check')
        else: check=False
        M0=Monomer()
        rotation_atomidxs=[int(sys.argv[3])]
        #rotation_list_list=[list(range(-90,90,5))]
        rotation_list_list=[list(range(-180,180,5))]
        M0.searchOptConf(gjffile,rotation_atomidxs,rotation_list_list,check=check,nnode=4,hpc='sr1',calctype='spd2')

    elif jobtype == 8:
        #(8) dimer opt
        g16cartopt(gjffile)

    elif jobtype == 9:
        #(9) crytal
        convert_crystal_gjf2xyz(gjffile)

    elif jobtype == 10:
    #(10) crystal opt
        ffname=sys.argv[3]
        opt_crystal_lmp(gjffile,ffname,nmollist=[4],natomlist=[29])

    elif jobtype == 11:
        ffname = sys.argv[3]
        npt(molname,ffname,supercell=[4,4,4])

    elif jobtype == 12:
        trjname = sys.argv[3]
        if len(sys.argv) == 6:
           nmol0 = int(sys.argv[4])
           natom0 = int(sys.argv[5])
           trj2gjf(molname,trjname,nmollist=[nmol0],natomlist=[natom0])
        elif len(sys.argv) == 8:
           nmol1 = int(sys.argv[6])
           natom1 = int(sys.argv[7])
           trj2gjf(molname,trjname,nmollist=[nmol0,nmol1],natomlist=[natom0,natom1])
        else: print('Illegal number of sys.argv. Check the syntax.')
        # python main_makeff.py 12 final_npt1800K mixsystem_npt1800K.trj 1080 29 1 67
        # python main_makeff.py 12 mixsystem_npt900K mixsystem_npt1200K.trj 864 28 1 67

    elif jobtype == 13:
        calcDimerG16_BSSE(molname)

    elif jobtype == 14:
        shutil.copyfile("/work/user/nakano/osc/mos/"+"mos/ext/gaussian/run.sh","./run.sh")

    elif jobtype == 15:
        #supercell(molname,supercell=[6,6,6],vacant=[0,0,4],shift=[0,0,4],ffname=sys.argv[3])
        supercell(molname,supercell=[1,1,1],vacant=[5,5,5],shift=[3,3,2],ffname=sys.argv[3])

    elif jobtype == 16:
        trjfile = sys.argv[3]
        change_trj_for_ovito(trjfile)

    elif jobtype == 17: # mix two ffs
        sysname = molname
        MM = MixMols()
        #MM.mix(sysname,ffnamelist=[sys.argv[3],sys.argv[4]],nmollist=[256,1])
        MM.mix(sysname,ffnamelist=[sys.argv[3],sys.argv[4]])

    else: print('Illegal jobtype!')



