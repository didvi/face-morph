import skimage as sk
import skimage.io as skio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from math import ceil
import os 

def show_triangulation(img, points):
    plt.triplot(matplotlib.tri.Triangulation(points.T[0], points.T[1]))
    plt.imshow(img)
    plt.show()

def show(imgs):
    if isinstance(imgs, list):
        n_row = 2
        n_col = ceil(len(imgs) / n_row)
        
        _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(img, cmap='gray')
        plt.show()
        return imgs
    else:
        # normalize
        imgs = imgs - np.min(imgs)
        imgs = imgs / max(1, np.max(imgs))
        imgs = sk.img_as_ubyte(imgs)
        skio.imshow(imgs)
        skio.show()
        return imgs

def toInt(img):
    img = img - np.min(img)
    img = img / max(1, np.max(img))
    img = sk.img_as_ubyte(img)
    return img

def save(img, imname, **kwargs):
    """Saves image in images folder with kwargs in image name as {key}_{value}
    """
    # normalize and convert image
    # img = img - np.min(img)
    # img = img / max(1, np.max(img))
    # img = sk.img_as_ubyte(img)

    fname = os.path.basename(imname)
    fname = [str(k) + '_' + str(kwargs[k]) for k in kwargs.keys()] + [fname]
    fname = "out/" + '_'.join(fname)
    skio.imsave(fname, img)
    return img
    
def read(img, color=True):
    # read in the image
    img = skio.imread(img, as_gray=not color)
    
    if not color:
        img = np.expand_dims(img, axis=2)

    # convert to double
    img = sk.img_as_float(img)
    return img

def crop(img, shape):
    """Center crops the image into specified shape"""
    if img.shape[0] != shape[0]:
        x = img.shape[0] - shape[0]
        img = img[x // 2 : x // 2 - x, :]
    
    if img.shape[1] != shape[1]:
        y = img.shape[1] - shape[1]
        img = img[ : , y // 2 : y // 2 - y]
    
    return img
