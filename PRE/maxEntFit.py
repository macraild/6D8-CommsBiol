from glob import glob

from scipy import optimize
import numpy as np
from matplotlib import pyplot

import extractPre

class maxEnt(object):
    def __init__(self, exptPre, calcPre, l=None):
        self.exptPre = exptPre
        self.calcPre = calcPre
        # Weight on the entropy term:
        self.l = l
        self.nSamples, self.length = self.calcPre.shape

    def chi2(self, weights=None):
        if weights is None:
            weights = self.result.x
        return sum(((self.exptPre[0,:]-np.dot(self.calcPre.T,weights))/self.exptPre[1,:])**2)/self.length

    def chi2Norm(self, weights=None):
        if weights is None:
            weights = self.result.x
        weights = weights/weights.sum()
        return self.chi2(weights)

    def entropy(self, weights):
        weights = weights/weights.sum() + 1e-200
        return -1*(weights*np.log(weights)).sum()

    def f(self, weights):
        return np.exp(-1*self.entropy(weights))

    def fit(self, initP=None, options={'maxfun':100000}):
        if initP is None:
            initP = np.ones(self.nSamples)/self.nSamples
        if self.l is None:
            l = self.chi2Norm(initP)/self.f(initP)
        else:
            l = self.l
        self.result = optimize.minimize(lambda p:l*self.f(p)+self.chi2Norm(p), 
                                   initP, bounds=self.nSamples*[(0,1),], options=options)

    def plot(self):
        pyplot.errorbar(range(self.length), self.exptPre[0,:], yerr=self.exptPre[1,:], marker='.',
                        linestyle='None')
        pyplot.plot(np.dot(self.calcPre.T, self.result.x))
