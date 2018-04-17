"""Extract PRE information from the output of the xplor calc.py script

(C) Chris MacRaild, Monash University, 2016-2018
"""

import sys
import numpy as np
import three2one

class PREReadError(Exception):
    pass

def extractPRE(fIn, proton='HN'):
    preArray = []
    S2radArray = []
    S2angArray = []
    residues = []
    for line in fIn:
        line = line.split()
        if not line:
            continue
        if line[:3] == ["StructureLoop:", "calculating", "structure"]:
            structure = int(line[3])
            preArray.append([])
            S2radArray.append([])
            S2angArray.append([])
            residues.append([])
            continue
        if len(line) > 2 and line[-1] == proton and line[-2] in three2one.map.keys():
            resid = int(line[-3])
            resname = line[-2]
            residues[-1].append(resname+' '+line[-3])
            continue
        if line[0] == "obs":
            pre = float(line[7])
            preArray[-1].append(pre)
            continue
        if line[0] == "S2angular:":
            S2angArray[-1].append(float(line[1]))
            continue
        if line[0] == "S2radial:":
            S2radArray[-1].append(float(line[1]))
            continue

    residues = np.array(residues)
    try:
        consistent = (residues == residues[0,:]).all()
    except IndexError:
        raise PREReadError, "Failed reading PREs from %s"%fIn.name
    if not consistent:
        raise PREReadError, "Inconsistent residues in %s"%fIn.name

    return np.array(preArray), np.array(S2angArray), np.array(S2radArray), residues[0,:]

def extractMulti(fileNames, proton='HN'):
    # Read first file to get data array size:
    with open(fileNames[0]) as fIn:
        pre, dump, dump, residues = extractPRE(fIn, proton)
    nStruct, nPre = pre.shape
    result = np.empty((nStruct*len(fileNames), nPre))
    for i,fName in enumerate(fileNames):
        with open(fName) as fIn:
            try:
                pre, dump, dump, residues = extractPRE(fIn, proton)
            except PREReadError as expt:
                print expt.msg
                continue
            try:
                result[nStruct*i:nStruct*(i+1),:] = pre
            except ValueError:
                print i, nStruct, fName
    return result

if __name__ == "__main__":
    fIn = open(sys.argv[1])
    try:
        proton = sys.argv[2]
    except IndexError:
        proton = 'HN'
    pre, dump, dump, residues = extractPRE(fIn, proton)
    avg = pre.mean(axis=0)
    std = pre.std(axis=0)
    for r,a,s in zip(residues, avg, std):
        print r,a,s
