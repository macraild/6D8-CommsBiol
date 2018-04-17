"""Model PREs in terms of a small number of localised probes

(C) Chris MacRaild, Monash University, 2016-2018
"""

import sys

import Bio.PDB
pdbParser = Bio.PDB.PDBParser()

import numpy as np
from scipy import optimize

class locateProbe(object):
    """Initiallised with a set of amide T2PREs (as I/Io ratios) and a
    corresponding structure, find the location of the electron spin 
    most consistent with the data.
    """
    def __init__(self, pre, pdbFileName, omega=600e6*2*np.pi, amideName=['HN']):
        """Find the location of the electron spin giving rise to pre,
        give the structure in pdbFile and the proton Larmor frequency
        omega.

        pre is an array of shape (N,2) containing T2PREs as I/Io ratios,
        with their associated uncertainties. 
        pdbFileName is the path to the reference structure. There must be 
        an ordered one-to-one corespondence between atoms with amideName in 
        the pdbFile, and PREs in pre.
        """
        self.pdbFile = pdbFileName
        self.pre = pre
        self.structure = pdbParser.get_structure('structure', pdbFileName)
        self.omega = omega
        coords = []
        for atom in self.structure.get_atoms():
            if atom.name in amideName:
                coords.append(atom.coord)
        self.amideCoords = np.array(coords)
        
    def calcPre(self, location, tau=2.):
        tau=tau*1e-9
        d6 = np.linalg.norm(self.amideCoords-location, axis=1)**-6
        t2 = 1./20*4.976e8**2*d6*(4*tau + 3*tau/(1+tau**2*self.omega**2))
        return np.exp(-1*t2/90)

    def Qfactor(self, location, tau=2.):
        """The PRE quality factor: sqrt(sum((exp-pred)**2)/sum(exp**2)),
        calculated over the T2 pre contribution (not I/Io)
        """
        if len(location) == 3:
            return np.sqrt(sum((np.log(self.pre[:,0])-np.log(self.calcPre(location, tau)))**2)/
                       sum(np.log(self.pre[:,0])**2))
        else:
            return np.sqrt(sum((np.log(self.pre[:,0])-np.log(self.calcMulti(location, tau)))**2)/
                       sum(np.log(self.pre[:,0])**2))

    def calcMulti(self, locations, tau=None):
        if tau is None:
            taus = locations[3::4]
            nProbes = len(locations)/4
        else:
            nProbes = len(locations)/3
            taus = None
        pre = np.empty((nProbes,self.pre.shape[0]))
        for x in xrange(nProbes):
            if taus is None:
                pre[x,:] = self.calcPre(locations[3*x:3*x+3],tau)
            else:
                pre[x,:] = self.calcPre(locations[4*x:4*x+3],locations[4*x+3])
        return np.exp(np.log(pre).mean(axis=0))


    def norm(self, location, tau=2.):
        """Return the variance weighted norm of the difference
        between experimental and predicted PREs, given lcation.
        """
        residual = self.pre[:,0]-self.calcPre(location, tau)
        return np.linalg.norm(residual/self.pre[:,1])

    def fit(self, tau=None):
        maxcoord = 1.2*self.amideCoords.max(axis=0)
        mincoord = 0.8*self.amideCoords.min(axis=0)
        initCoords = mincoord + np.random.uniform(size=3)*(maxcoord-mincoord)
        pInit = initCoords.tolist()
        if tau is None:
            pInit.append(2.)
            func = lambda p: self.norm(p[:3],p[3])
        else:
            func = lambda p: self.norm(p,tau)
        #self.fitResult = optimize.basinhopping(func, pInit, niter=1000,
        #                         stepsize=10,
        #                         minimizer_kwargs={'method':'Powell'})
        self.fitResult = optimize.minimize(func, pInit, method='Powell')

    def fitMulti(self, nProbes, tau=None, fitQ=False):
        maxcoord = 1.2*self.amideCoords.max(axis=0)
        mincoord = 0.8*self.amideCoords.min(axis=0)
        pInit = []
        for x in xrange(nProbes):
            initCoords = mincoord + np.random.uniform(size=3)*(maxcoord-mincoord)
            pInit.extend(initCoords.tolist())
            if tau is None:
                pInit.append(2.)
        if fitQ:
            self.fitResult = optimize.basinhopping(self.Qfactor, pInit, niter=100,
                                 stepsize=10,
                                 minimizer_kwargs={'method':'Powell','args':(tau)})
            return
        pre = np.empty((nProbes,self.pre.shape[0]))
        def func(p):
            for x in xrange(nProbes):
                if tau is None:
                    pre[x,:] = self.calcPre(p[4*x:4*x+3],p[4*x+3])
                else:
                    pre[x,:] = self.calcPre(p[3*x:3*x+3],tau)
            residual = self.pre[:,0] - np.exp(np.log(pre).mean(axis=0))
            return np.linalg.norm(residual/self.pre[:,1])
        self.fitResult = optimize.basinhopping(func, pInit, niter=100,
                                 stepsize=10,
                                 minimizer_kwargs={'method':'Powell'})

