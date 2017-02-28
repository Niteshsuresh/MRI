import sys, os, numpy as np
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from skimage import measure
from skimage import segmentation
from skimage import io, filters
from skimage import feature
from scipy import ndimage as ndi
from skimage.morphology import watershed


ic = io.imread_collection(sys.path[0]+os.sep+'subj1'+os.sep+'*.gif');
print(len(ic))
f, axes = plt.subplots(nrows=4, ncols=len(ic), figsize=(15, 10))

for i, image in enumerate(ic):
    axes[0,i].imshow(image, cmap="gray")
    axes[0,i].axis('off')

for i, image in enumerate(ic):
    #val = filters.threshold_otsu(image)
    edges = filters.sobel(image)
    val = filters.threshold_otsu(edges)
    non_edges = edges < val
    #clean_border = segmentation.clear_border(non_edge)
    axes[1,i].imshow(non_edges,cmap="gray")
    axes[1,i].axis('off')

#for i, image in enumerate(ic):
#    val = filters.threshold_otsu(image)
#    edge = image < val
#    axes[2,i].imshow(edge, cmap="gray")
#    axes[2,i].axis('off')

for i, image in enumerate(ic):
    val = filters.threshold_otsu(image)
    edge = image < val
    #clean_border = segmentation.clear_border(edge)
    #all_labels = measure.label(clean_border)
    #print(all_labels.shape)
    axes[2,i].imshow(edge, cmap="gray")
    axes[2,i].axis('off')

for i, image in enumerate(ic):
    edges = filters.sobel(image)
    threshold = filters.threshold_otsu(edges)
    no_edge = edges < threshold
    distance_from_edge = ndi.distance_transform_edt(no_edge)
    #clean_border = segmentation.clear_border(edge)
    #all_labels = measure.label(no_edge, background=0)
    peaks = feature.peak_local_max(distance_from_edge, min_distance=20)
    peaks_image = np.zeros(image.shape, np.bool)
    peaks_image[tuple(np.transpose(peaks))] = True
    seeds, num_seeds = ndi.label(peaks_image)
    print(num_seeds)
    ws = watershed(edges, seeds)
    from skimage import color
    #print(all_labels.shape)

    v = color.label2rgb(ws , image)
    #print(v)
    axes[3,i].imshow(color.label2rgb(ws , image))
    axes[3,i].axis('off')
#plt.imshow(image)
#plt.show()
#print(type(ic), ic)
