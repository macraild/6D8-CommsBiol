"""MDAnalysis methods for contact analysis.

(C) Chris MacRaild, Monash University. 2016-2017
"""

import numpy as np
from MDAnalysis.analysis import contacts as analysisContacts

def contactMap(u, atomSelA, atomSelB, radius=7.0):
    """Maps the frequency of contacts between selected atoms
    over the trajectory."""
    
    atomGroupA = u.atoms.select_atoms(atomSelA)
    atomGroupB = u.atoms.select_atoms(atomSelB)

    da = np.zeros((len(atomGroupA), len(atomGroupB)))
    for n,ts in enumerate(u.trajectory):
        da = (da*n + (analysisContacts.distance_array(atomGroupA.positions,
              atomGroupB.positions)<radius).astype(float))/(n+1)

    return da

def minTS(u, atomSelA, atomSelB, seqDist=2, stride=1):
    """Return timeseries of the minimum distance between atomGroups
    A and B. 

    Where A and B share common protein chains, atoms must be > seqDist 
    residues apart in sequence.
    """

    atomGroupA = u.atoms.select_atoms(atomSelA)
    atomGroupB = u.atoms.select_atoms(atomSelB)

    minTS = []
    for fr in u.trajectory[::stride]:
        dists = []
        for a in atomGroupA:
            for b in atomGroupB:
                if a.segment is b.segment:
                    if np.abs(a.resnum-b.resnum) <= seqDist:
                        continue
                dists.append(np.linalg.norm(a.position-b.position))
        minTS.append(min(dists))

    return minTS

def numTS(u, atomSelA, atomSelB, radius=7.0, seqDist=2, stride=1):
    """Return timeseries of the number of distance<radius between atomGroups
    A and B. 

    Where A and B share common protein chains, atoms must be > seqDist 
    residues apart in sequence.
    """

    atomGroupA = u.atoms.select_atoms(atomSelA)
    atomGroupB = u.atoms.select_atoms(atomSelB)

    numTS = []
    for fr in u.trajectory[::stride]:
        count = 0
        for a in atomGroupA:
            for b in atomGroupB:
                if a.segment is b.segment:
                    if np.abs(a.resnum-b.resnum) <= seqDist:
                        continue
                if np.linalg.norm(a.position-b.position) < radius:
                    if a in atomGroupB:
                        count += 0.5
                    else:
                        count += 1
        numTS.append(count)

    return numTS

def averageLifetime(timeseries, radius=7.0):
    """Return the average contact lifetime for each atom-pair in 
    timeseries.
    """

    avgLifetimes = {}
    for k,v in timeseries.items():
        lifetimes = []
        contacts = (np.array(v)<radius).astype(int)
        current = 0
        for c in contacts:
            if c == 0 and current:
                lifetimes.append(current)
                current=0
            else:
                current += c
        avgLifetimes[k] = lifetimes
    return avgLifetimes

def contactLifetime(u, atomSelA, atomSelB, radius=7.0):
    """Return the lifetime of each contact between atoms A and B.

    This counts a new contact for each time the contact is broken
    and remade.
    """

    currentContacts = {}
    atomGroupA = u.atoms.select_atoms(atomSelA)
    atomGroupB = u.atoms.select_atoms(atomSelB)
    lifetimes = []

    for fr in u.trajectory:
        for a in atomGroupA:
            for b in atomGroupB:
                if np.linalg.norm(a.position-b.position) < radius:
                    if not (a,b) in currentContacts:
                        # register start time for new contact
                        currentContacts[(a,b)] = fr.time
                else:
                    if (a,b) in currentContacts:
                        # contact now broken
                        lifetimes.append(fr.time-currentContacts[(a,b)])
                        del currentContacts[(a,b)]

    return lifetimes

def timeSeries(u, atomSelA, atomSelB, radius=7.0, freq=1e-2):
    """Return timeseries of atom distances for all pairs of atoms
    in A and B that spend more than freq within radius."""

    atomGroupA = u.atoms.select_atoms(atomSelA)
    atomGroupB = u.atoms.select_atoms(atomSelB)

    ts = {}
    cm = contactMap(u, atomSelA, atomSelB, radius)
    for i in xrange(cm.shape[0]):
        for j in xrange(cm.shape[1]):
            if cm[i,j] >= freq:
                ts[(i,j)] = []

    for fr in u.trajectory:
        for i,j in ts:
            ts[(i,j)].append(np.linalg.norm(atomGroupA[i].position - 
                                            atomGroupB[j].position))

    return ts
