from skimage import io
import numpy as np
import os, sys
from matplotlib import pyplot as plt
from scipy.linalg import norm
from scipy.ndimage import rotate


def load_processed_images(base_path):
    ic = io.imread_collection(base_path+'*.gif')
    return ic
def extract_grey(image):
    return image == 127
def extract_white(image):
    return image == 191
def extract_csf(image):
    return image == 63

def process(base_path):
    ic = load_processed_images(base_path)
    features = []
    for i, img in enumerate(ic):
        g_img = extract_grey(img)
        w_img = extract_white(img)
        c_img = extract_csf(img)
        g_mean = mean(g_img)
        w_mean = mean(w_img)
        c_mean = mean(c_img)
        g_s_ud = symmetry_upTdown(g_img)
        g_s_lt = symmetry_lefTright(g_img)
        w_s_ud = symmetry_upTdown(w_img)
        w_s_lt = symmetry_lefTright(w_img)
        c_s_ud = symmetry_upTdown(c_img)
        c_s_lt = symmetry_lefTright(c_img)
        features.append([g_mean, w_mean, c_mean, g_s_ud, g_s_lt, w_s_lt, w_s_ud, c_s_lt, c_s_ud])

    return features

def mean(image):
    return np.mean(image)

def plot_preprocessed_img(image):
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    grey_img = extract_grey(image)
    white_img = extract_white(image)
    csf_img = extract_csf(image)
    axes[0,0].imshow(grey_img, cmap="gray")
    axes[0,0].axis('off')
    axes[0,1].imshow(white_img, cmap="gray")
    axes[0,1].axis('off')
    axes[0,2].imshow(csf_img, cmap="gray")
    axes[0,2].axis('off')
    plt.show()

def symmetry_upTdown(img):
    r_img = rotate(img, 90, reshape=False)
    fliplrimg = np.fliplr(r_img)
    arry = img - fliplrimg
    #flatten the array & sum the values
    return norm(arry.ravel(), 0)

def symmetry_lefTright(image):
    fliplrimg = np.fliplr(image)
    arry = image - fliplrimg
    return norm(arry.ravel(), 0)

#dementia
dementia_base_path = sys.path[0]+ os.sep + 'dementia'+os.sep +'subj' + os.sep
non_dementia_base_path = sys.path[0]+ os.sep + 'Non_dementia'+os.sep +'subj' + os.sep
print(process(dementia_base_path))
print(process(non_dementia_base_path))
