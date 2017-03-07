import numpy as np
import os, sys
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
no_dementia_base_path = sys.path[0]+os.sep+'Non_dementia'
dementia_base_path = sys.path[0]+os.sep+'dementia'

base_path = dementia_base_path
for id in range(1,13):
    os.mkdir(base_path+os.sep+str(id))
    #f, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    sarraster = plt.imread(base_path+os.sep+'subj'+os.sep+str(id)+'.gif')
    # Removing speckles
    sarraster = ndi.median_filter(sarraster , size=2)
    # Flatten image to get line of values
    flatsarraster = sarraster.flatten().astype(float)

    print(flatsarraster.shape)

    # In remaining subplots add k-means classified images
    for i in range(2,6):
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
        print(sarraster.shape)
        #Since code contains the classified values, reshape into SAR dimensions
        codeim = code.reshape(sarraster.shape[0], -1)
        print(codeim.shape)
        #codeim = ndi.median_filter(codeim , size=4)
        for j in range(i):
            #Plot the subplot with (i+2)th k-means
            ax = plt.subplot(2,4,j+2)
            c = codeim == j
            c = ndi.median_filter(c , size=10)
            plt.axis('off')
            xlabel = str(j+1) , ' clusters'
            ax.set_title(xlabel)
            plt.imsave(base_path+os.sep+str(id)+'/gradient'+str(i)+str(j)+'.png', c, cmap='gray')

    #plt.show()
