import numpy as np
import argparse
import glob
import matplotlib as plt
import skimage.transform

from morph import *
from helpers import *

# Helper functions for reading data
def get_by_face_type(type=1, file_ext='.asf'):
    """Gets list of files from imm face db with correct face type

    Args:
        type (int, optional): Type number. Defaults to 1.

    Returns:
        [list]: list of file paths
    """
    files = glob.glob(f'in/imm_face_db/*-{type}m{file_ext}')
    files.sort()
    return files

def read_asf(file):
    """Reads x, y points from asf file

    Args:
        file (str): path to asf file

    Returns:
        np.ndarray, np.ndarray: list of x points, list of y points
    """
    data = np.genfromtxt(file, skip_header=16, skip_footer=1, usecols=(2, 3))
    return data[:, 0], data[:, 1]

def calc_avg_points(files):
    avg_x, avg_y = read_asf(files.pop())
    for f in files:
        x, y = read_asf(f)
        avg_x = np.mean([avg_x, x], axis=0)
        avg_y = np.mean([avg_y, y], axis=0)
    return avg_x, avg_y

def format_points(x, y, width, height):
    """Unnormalizes points, Adds corner points, and Formats as a numpy array

    Args:
        x ([arr])
        y ([arr])
    
    Returns:
        np.ndarray with size + 4
    """
    x, y = x * width, y * height
    x = np.append(x, [1, 1, width - 1, width - 1])
    y = np.append(y, [1, height - 1, 1, height - 1])
    return np.column_stack((x, y))


# Morph functions
def morph_to_points(img1, orig_points, target_points):
    """Morphs the image from original keypoints to target keypoints

    Args:
        img (np.ndarray): original image
        orig_points (np.ndarray): keypoints in original image
        target_points (np.ndarray): keypoint locations after morphing
    """
    morphed_img = np.zeros(img1.shape)

    # compute affine transformations
    for t in Delaunay(target_points).simplices:
        first_tri = np.array([orig_points[i] for i in t])
        avg_tri = np.array([target_points[i] for i in t])

        # compute affine transformation matrix
        transform1 = compute_transform(first_tri, avg_tri)

        # compute indices needed
        rr, cc = sk.draw.polygon(avg_tri.T[1], avg_tri.T[0])

        # compute indices needed in image
        first_indices = np.column_stack((cc, rr, np.ones(rr.shape[0])))
        first_indices = transform1 @ first_indices.T

        # shit so it works
        def make_indexable(arr):
            arr[0] = (np.clip(arr[0], 0, img1.shape[0] - 1))
            arr[1] = (np.clip(arr[1], 0, img1.shape[1] - 1))
            return arr.astype(int)
        
        first_indices = make_indexable(first_indices)
        
        # find pixels
        morph = img1[first_indices[1], first_indices[0]] 
        morphed_img[rr, cc] = morph

    return toInt(morphed_img)

def avg_morph(img_paths, point_files, avg_points):
    height, width, c = read(img_paths[0]).shape

    mean_morphed_img = np.zeros((height, width, c))
    for i, p in zip(img_paths, point_files):
        img = read(i)
        x, y = read_asf(p)

        # format points
        points1 = format_points(x, y, width, height)

        # morph image to average points
        morphed_img = morph_to_points(img, points1, avg_points)
        
        # add to mean image
        mean_morphed_img += morphed_img

    mean_morphed_img = mean_morphed_img / len(img_paths)
    return mean_morphed_img

def main(args):
    # get file names
    files = get_by_face_type(type=args.type)
    img_paths = get_by_face_type(type=args.type, file_ext='.jpg')

    # read data and calculate average points
    height, width = read(img_paths[0]).shape[:2]
    avg_x, avg_y = calc_avg_points(files)
    avg_points = format_points(avg_x, avg_y, width, height)

    # create average image
    avg_morphed_img = avg_morph(img_paths, files, avg_points)
    
    show(avg_morphed_img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", default=1)
    args = ap.parse_args()

    main(args)

