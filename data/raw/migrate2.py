import numpy as np
import os, sys
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

#f, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
sarraster = plt.imread(sys.path[0]+os.sep+'subj1/3.gif')
# Removing speckles
sarraster = ndi.median_filter(sarraster , size=6)
# Flatten image to get line of values
flatsarraster = sarraster.flatten().astype(float)
# In remaining subplots add k-means classified images
for i in range(2,7):

    #This scipy code classifies k-mean, code has same length as flattened
    #SAR raster and defines which class the SAR value corresponds to
    centroids, variance = kmeans(flatsarraster, i)
    code, distance = vq(flatsarraster, centroids)
    fig = plt.figure()
    fig.suptitle('K-Means Classification')

    # In first subplot add original SAR image
    ax = plt.subplot(241)
    plt.axis('off')
    ax.set_title('Original Image')
    plt.imshow(sarraster, cmap = 'gray')

    #Since code contains the classified values, reshape into SAR dimensions
    codeim = code.reshape(sarraster.shape[0], sarraster.shape[1])
    #codeim = ndi.median_filter(codeim , size=4)
    for j in range(i):
        #Plot the subplot with (i+2)th k-means
        ax = plt.subplot(2,4,j+2)
        plt.axis('off')
        xlabel = str(j+1) , ' clusters'
        ax.set_title(xlabel)
        plt.imshow(codeim == j, cmap='gray')

plt.show()
