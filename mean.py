import numpy as np
import argparse
import glob
import matplotlib as plt

from morph import *
from helpers import *

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

def avg_points(files, width, height):
    avg_x, avg_y = read_asf(files.pop())
    for f in files:
        x, y = read_asf(f)
        avg_x = np.mean([avg_x, x], axis=0)
        avg_y = np.mean([avg_y, y], axis=0)
    return avg_x * width, avg_y * height

def main(args):
    files = get_by_face_type(type=args.type)
    img_paths = get_by_face_type(type=args.type, file_ext='.jpg')

    height, width = read(img_paths[0]).shape[:2]
    avg_x, avg_y = avg_points(files, width=width, height=height)
    
    if args.show:
        plt.triplot(matplotlib.tri.Triangulation(avg_x, avg_y))
        plt.imshow(read(img_paths[args.img_index]))
        plt.show()

    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", default=1)
    ap.add_argument('-s', '--save', type=bool, default=True)
    ap.add_argument('-v', '--video', type=bool, default=False)
    ap.add_argument('-i', '--img_index', default=0)
    ap.add_argument('-s', '--show', type=bool, default=True, help='Show average triangulation or not')
    args = ap.parse_args()

    main(args)

