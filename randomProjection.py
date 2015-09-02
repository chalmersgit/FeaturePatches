"""
Random Projection
Author: Andrew Chalmers, 2015

Reduces the dimensionality of feature vectors using random projection.
Feature vectors are obtained from patches of a matches using the patches module

References:

Johnson-Lindenstrauss Random Projection
2009, S. Kakade and G. Shakhnarovich
http://ttic.uchicago.edu/~gregory/courses/LargeScaleLearning/lectures/jl.pdf

"Database-friendly random projections: Johnson-Lindenstrauss with binary coins" 
2002, D. Achlioptas
https://users.soe.ucsc.edu/~optas/papers/jl.pdf

"Texture Classification from Random Features"
2012, L. Liu and W. Fieguth)
http://vip.uwaterloo.ca/files/publications/Paper1JournalLiFieguth_final.pdf

Stackoverflow:
http://stackoverflow.com/questions/7474508/random-projection-algorithm-pseudo-code
"""

import numpy as np
import math

import patches as pa 

def randomProjection(X, verbose = False):
	"""
	X should be of shape (N,M)
	Where N is the number of features 
	and M is the number of dimensions 
	"""
	numFeats 	= X.shape[0]
	numDims 	= X.shape[1]

	k			= (20 * np.log(numFeats)) / (math.e*math.e)
	reducedDims	= int(k)
	norm 		= 1.0 / (np.sqrt(k))

	randMat		= (np.ones((numDims,reducedDims))*np.clip(1, -1, np.random.normal(0, 1, (numDims,reducedDims)))).astype(int)
	X_reduced	= norm*np.dot(X,randMat)
	
	if verbose:
		print "Original dimensions:\t", X.shape
		print "Reduced dimensions:\t", X_reduced.shape

	return X_reduced

if __name__ == "__main__":
	print 'Random Projection'
	width 		= 200
	height 		= 200

	N 			= int(width*height)		# total elements
	patchSize 	= int(np.sqrt(width))	# square patch resolution using width

	arr = np.arange(N).reshape(height,width)

	print 'Width x Height:\t', width,'x',height
	print 'Total elements:\t',N
	print 'Patch size:\t', patchSize
	print

	patches = pa.getPatches(arr, (patchSize,patchSize), patchSize)
	features = pa.transToFeatVecs(patches)

	featuresProj = randomProjection(features, True)

