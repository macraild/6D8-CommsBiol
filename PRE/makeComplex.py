"""Use Biopython and Xplor-nih to build flexiblemeccano-derived MSP2 peptide 
conformations into a complex with 6D8 and model nitroxide conformations.

(C) Chris MacRaild, Monash University - 2016-2018

Instructions for use:

    Run FexibleMeccano to generate a suitable starting ensemble of your
    disordered partner

    Edit the function makeStruct according to the requirements of your system

    Ensure all the scripts from [xplor-distr]/eginput/pre/fitting are
    in cwd, and: 
    - Change the top of buildtag.py to include:

        import sys
        protocol.loadPDB(sys.argv[1])
    
      in place of the existing loadPDB() call.

    - Change calc.py to include:
      
        # Setup of backbone 1H PRE restraints  
        import sys
        pre = create_PREPot("PRE",sys.argv[1],"normal")

      in place of the existing create_PREPot() call.

    - Modify rand.py to calculate one structure.

    - Make any other necessary system-specific changes to these scripts

    Run this file as:

    > python makeComplex.py preFile [nProcs] flexiblemeccanoFiles

    where: 
    preFile contains the experimental PREs in xplor format
    nProcs is the number of processors to run (4)

    eg:

    > python makeComplex.py pre.tbl 4 ./flexiblemeccano/*.pdb
"""

import os, tempfile, subprocess, shutil, glob
from contextlib import contextmanager

import Bio.PDB
pdbParser = Bio.PDB.PDBParser()
pdbIO = Bio.PDB.PDBIO()
pdbl = Bio.PDB.PDBList(server='http://www.rcsb.org/pdb/files',
           pdb="/vcp1/people/macraild/MSP2/epitopePredictions/structures")
align = Bio.PDB.Superimposer()

xplorBin = "/vcp1/people/macraild/progs/xplor-nih-2.45/bin/xplor"

class excludeHetatm(Bio.PDB.Select):
    def accept_atom(self, atom):
        if atom.get_full_id()[3][0] == ' ':
            return 1
        return 0

class AtomClash(Exception):
    def __init__(self, atoms):
        self.atoms = atoms

copy=Bio.PDB.Atom.copy
def myCopy(self):
    shallow = copy.copy(self)
    for child in self.child_dict.values():
        shallow.disordered_add(child.copy())
    return shallow
Bio.PDB.Atom.DisorderedAtom.copy=myCopy

def makeStruct(pdbMSP2, fOut):
    """Make a complex between flexiblemeccano structure pdbMSP2, and 4qyo, and output 
    to fOut.

    This function is highly customised to the MSP2/6D8 system, and will need to be 
    re-written to apply this approach to a different system.
    """
    # Load structures:
    struct6D8 = pdbParser.get_structure('6D8', '/vcp1/people/macraild/MSP2/scFv/6D8/4qyo.pdb')
    structMSP2 = pdbParser.get_structure('MSP2', pdbMSP2)

    # Align MSP2 to 6D8:
    align6D8 = []
    alignMSP2 = []
    residueOffset = 13
    for res in range(14,23):
        align6D8.extend([struct6D8[0]['Q'][res]['N'],
                         struct6D8[0]['Q'][res]['CA'],
                         struct6D8[0]['Q'][res]['C']])
        alignMSP2.extend([structMSP2[0][' '][res-residueOffset]['N'],
                          structMSP2[0][' '][res-residueOffset]['CA'],
                          structMSP2[0][' '][res-residueOffset]['C']])
    align.set_atoms(align6D8, alignMSP2)
    align.apply(structMSP2.get_atoms())

    # Search for clashes between aligned flexiblemeccano model and 6D8:
    atoms6D8=[]
    for atom in struct6D8[0]['A'].get_atoms():
        if atom.parent.id[0] == ' ':
            atoms6D8.append(atom)
    for atom in struct6D8[0]['B'].get_atoms():
        if atom.parent.id[0] == ' ':
            atoms6D8.append(atom)
    search  = Bio.PDB.NeighborSearch(atoms6D8)
    for res in range(10,22):
        atom = structMSP2[0][' '][res]['CA']
        result = search.search(atom.coord, 3.0)
        if result:
            raise AtomClash((atom, result))

    # Assemble complex as a Bio.PDB.Structure object
    compl = Bio.PDB.Structure.Structure('compl')
    model = Bio.PDB.Model.Model(0)
    compl.add(model)
    model.add(struct6D8[0]['A'].copy())
    model.add(struct6D8[0]['B'].copy())
    MSP2 = structMSP2[0][' '].copy()
    MSP2.id = 'Q'
    MSP2[21].resname = 'CYSP'
    model.add(MSP2)

    # Write complex to pdb file
    pdbIO.set_structure(compl)
    pdbIO.save(fOut, select=excludeHetatm())
    fOut.flush()
    # Fix error in Biopython PDB output when resname has 4 letters:
    subprocess.call(['sed', '-i', '-e', 's/CYSP /CYSP/g', fOut.name])
            
def runXplor(pdbName, outName, preName):
    """Use the scripts in [xplor-distr]/eginput/pre/fitting to build the 
    nitroxide probe in 5 configurations.

    Ensure all the scripts from [xplor-distr]/eginput/pre/fitting are
    in cwd, and: 
    - Change the top of buildtag.py to include:

        import sys
        protocol.loadPDB(sys.argv[1])
    
      in place of the existing loadPDB() call.

    - Change calc.py to include:
      
        # Setup of backbone 1H PRE restraints  
        import sys
        pre = create_PREPot("PRE",sys.argv[1],"normal")

      in place of the existing create_PREPot() call.

    - Modify rand.py to calculate one structure.

    - Make any other necessary system-specific changes to these scripts
    """
    subprocess.call([xplorBin, '-py', 'buildtag.py', '-o', 'buildtag.out', pdbName])
    subprocess.call([xplorBin, 'make5conf.inp', '-o', 'make5conf.out'])
    subprocess.call([xplorBin, '-py', 'rand.py', '-o', 'rand.out'])
    subprocess.call([xplorBin, '-py', 'calc.py', '-o', outName, preName])

@contextmanager
def workingDir(outputDir=None, cleanupFiles=None):
    """Setup and teardown of a temporary working directory for xplor runs.
    If outputDir is defined, rename the output into that directory, after 
    first deleting any files matched by glob(cleanupFiles).
    """
    #setup
    cwd = os.getcwd()
    workingDir = tempfile.mkdtemp()
    for fName in ['buildtag.py', 'make5conf.inp', 'rand.py', 'fit.py', 'calc.py']:
        os.symlink(os.path.join(cwd, fName), os.path.join(workingDir, fName))
    os.chdir(workingDir)

    try:
        yield workingDir
    finally:
        #teardown
        if not cleanupFiles is None:
            try:
                for file in glob.glob(os.path.join(workingDir,cleanupFiles)):
                    os.remove(file)
            except TypeError:
                for cfGlob in cleanupFiles:
                    for file in glob.glob(os.path.join(workingDir,cfGlob)):
                        os.remove(file)
        os.chdir(cwd)
        for fName in ['buildtag.py', 'make5conf.inp', 'rand.py', 'fit.py', 'calc.py']:
            os.remove(os.path.join(workingDir,fName))
        try:
            if not outputDir is None:
                shutil.move(workingDir, outputDir)
        except Exception as e:
            print e
            print "FAILED moving working dir ouput. Temp dir retained:"
            print workingDir
        else:
            try:
                shutil.rmtree(workingDir)
            except OSError:
                pass

def run(inFileName, preTblName='3D7_pre.tbl'):
    cwd = os.getcwd()
    preTblPath = os.path.join(cwd,preTblName)
    outputDir = os.path.join(cwd,os.path.splitext(inFileName)[0])
    with workingDir(outputDir, ['rand*','tagged*']):
        fOut = open('test.pdb','w')
        try:
            makeStruct(os.path.join(cwd,inFileName), fOut)
        except AtomClash:
            return
        fOut.close()
        runXplor('test.pdb', os.path.join(cwd,inFileName+'_pre.dat'), preTblPath)


if __name__ == "__main__":
    import multiprocessing, sys
    preFile = sys.argv[1]
    files = []
    nextArg = sys.argv[2]
    try:
        nProc = int(nextArg)
    except ValueError:
        nProc = 4
        files.append(nextArg)
    for fName in sys.argv[3:]:
        # If calculated PREs already exist for this input, skip
        if not os.path.isfile(fName+'_pre.dat'):
            files.append(fName)
    pool = multiprocessing.Pool(nProc)
    pool.map(run, files)
  
