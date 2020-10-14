# This code adapted from some code Omer Mano wrote for Matlab

import numpy as np

def scatterBar(Ys):
    if type(Ys) is list:
        # list stuff here... I think I might just delete this...
        pass
    else:
        try:
            numPoints, numBars = Ys.shape
        except ValueError:
            # the second dimension of the array wasn't passed, so it's one
            Ys = Ys[:, None]
            numPoints, numBars = Ys.shape

    XYPos = np.zeros((numPoints,2,numBars))
    indScrambles = np.zeros((numPoints,numBars))
    for ii in range(numBars):
        XYPos[:,:,ii] = scatterPoints(Ys[:,ii],Ys.max() - Ys.min()) - np.array([ii,0])
        indsToSortInput = np.argsort(Ys[:,ii], axis=0)
        # I feel like I'm missing something and being dumb here, but sorting
        # the inds to sort the array tells you how to unsort the array to its
        # original position
        indScrambles[:, ii] = np.argsort(indsToSortInput).astype('int')


    return XYPos, indScrambles 

def scatterPoints(Ys,plotRange):

    numPoints = Ys.shape[0]
    XYPos = np.concatenate((np.zeros((numPoints,1)),np.sort(Ys)[:,None]), axis=1)

    XYPos[:,0] = XYPos[:,0] + 1*np.random.randn(numPoints)

    for jj in [1,2]:
        kickMagnitude = 0.4**jj # yay magic numbers... but this works
        XYPos[:,0] = XYPos[:,0] + 0*kickMagnitude*np.random.randn(numPoints)
        scrambledPoints = np.random.permutation(np.arange(numPoints))
        if (numPoints % 2) == 0:
            halfPoints = numPoints/2
        else:
            halfPoints = np.floor(numPoints/2)
        
        scramblePick = scrambledPoints[:int(halfPoints)]
        # the halfPoints+halfPoints ensures they're both halfPoints length and
        # also that they're not overlapping
        otherPick = scrambledPoints[int(halfPoints):int(halfPoints+halfPoints)] 
        
        temp = XYPos[scramblePick,0].copy()
        XYPos[scramblePick,0] = XYPos[otherPick,0]
        XYPos[otherPick,0] = temp
        for ii in range(20):
            forces = generateEnergies(XYPos,plotRange)
            # wee magic numbers, but hey it works
            XYPos[:,0] = XYPos[:,0] + kickMagnitude*(0.95**(ii+1))*(4*np.tanh(forces/4))

    XYPos[:,0] = 2*XYPos[:,0]
    return XYPos 

def generateEnergies(XYPos,plotRange):

    numPoints = XYPos.shape[0]
    centeringTerm = -XYPos[:,0]
    interactionTerm = np.zeros((numPoints))
    # wee magic numbers, but hey it works
    yModifier = 10/plotRange

    for ii in range(numPoints):
        #         testStart = max(1,ii-5)
        #         testEnd = min(numPoints,ii+5)
        pairwiseVectors = XYPos[ii,:] - XYPos
        pairwiseVectors[:,1] = pairwiseVectors[:,1]*yModifier
        pairwiseDistancesSquared = np.sum(pairwiseVectors**2,axis=1) 
        #         xDirModifier = pairwiseVectors(:,1)./sqrt(pairwiseDistancesSquared)
        #         xTermContributions = xDirModifier./(pairwiseDistancesSquared.^2)
        #         xTermContributions(ii) = 0
        #         xTermContributions = tanh(xTermContributions)
        #         interactionTerm(ii) = tanh(sum(xTermContributions))
        #         collided = pairwiseDistancesSquared < 0.1
        #         pushDirection = tanh(sum(sign(pairwiseVectors(collided,1))))
        pushDirection = np.nansum(np.tanh(np.sign(pairwiseVectors[:,0])/(1000*pairwiseDistancesSquared)))
        interactionTerm[ii] = pushDirection

    xForces = centeringTerm + 0.1*interactionTerm

    return xForces
