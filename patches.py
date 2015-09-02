"""
Patch Extraction
Author: Andrew Chalmers, 2015

References:

"Texture Classification from Random Features"
2012, L. Liu and W. Fieguth)
http://vip.uwaterloo.ca/files/publications/Paper1JournalLiFieguth_final.pdf
"""

import numpy as np
from skimage.util.shape import view_as_windows

def getPatches(arr, patchSize, step=1):
	return view_as_windows(arr, patchSize, step)

def transToFeatVecs(patches):
	return patches.reshape(getNumPatches(patches),getNumPatchElements(patches))

def getNumPatches(patches):
	return patches.shape[0]*patches.shape[1]

def getNumPatchElements(patches):
	return patches.shape[2]*patches.shape[3]

if __name__ == "__main__":
	print 'Patches'
	width 		= 200
	height 		= 200

	N 			= int(width*height)		# total elements
	patchSize 	= int(np.sqrt(width))	# square patch resolution using width

	arr = np.arange(N).reshape(height,width)

	print 'Width x Height:\t', width,'x',height
	print 'Total elements:\t',N
	print 'Patch size:\t', patchSize
	print

	patches = getPatches(arr, (patchSize,patchSize), patchSize)
	features = transToFeatVecs(patches)

	print patches.shape
	print features.shape

