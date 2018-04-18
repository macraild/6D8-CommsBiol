"""Find pockets in an MDAnalysis density grid using a LigSite
like algorithm.

(C) Chris MacRaild, Monash University 2017-2018"""

import numpy as np
from MDAnalysis.analysis import density

def pocketGrid(density, universe, atomSel="all"):
    """For an MDAnalysis density and the coresponding universe,
    return an isomorphous grid populated with pocket scores."""

    coords = universe.selectAtoms(atomSel).coordinates()
    nAtoms = coords.shape[0]
    directions = [[1,0,0],
                  [0,1,0],
                  [0,0,1],
                  [1,1,1],
                  [-1,1,1],
                  [1,-1,1],
                  [1,1,-1]]
    grid = np.zeros(density.grid.shape,dtype=np.int8).flatten()

    for i,point in enumerate(density.centers()):
        # Point too close to atoms (ie. is inside the protein)
        if np.linalg.norm(point-coords, axis=1).min()<3.1:
            continue
        for d in directions:
            # distance from each atom to the line from point along d:
            vecs = (point-coords)-(np.dot((point-coords),d)*np.array([d,]*nAtoms).T).T
            dists = np.linalg.norm(vecs, axis=1)
            # the projection from point along d for each bisected atom:
            proj = np.dot((point-coords),d)[np.where(dists<3.1)]
            if np.signbit(proj).all():
                # proj all positive, so not in pocket along d
                continue
            if np.signbit(proj).any():
                # in pocket along d, as proj has both +ve and -ve members
                grid[i] += 1
                continue
            # otherwise proj all -ve, so not in pocket

    return grid.reshape(density.grid.shape)

