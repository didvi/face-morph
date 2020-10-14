import numpy as np
import argparse
import glob
import matplotlib as plt
import skimage.transform

from morph import *
from helpers import *
from mean import *


def main(args):
    # get file names
    files = get_by_face_type(type=args.type)
    img_paths = get_by_face_type(type=args.type, file_ext='.jpg')

    # read data 
    img = read(img_paths[args.img_index])
    height, width = img.shape[:2]
    x, y = read_asf(files[args.img_index])
    points = format_points(x, y, width, height)

    # calculate average points
    avg_x, avg_y = calc_avg_points(files)
    avg_points = format_points(avg_x, avg_y, width, height)

    # create average image
    morphed_img = morph_to_points(img, points, avg_points)
    create_video(img, morphed_img, points, avg_points, 'out/guy_to_average')
    show(morphed_img)
    show_triangulation(img, points)
    show_triangulation(morphed_img, avg_points)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", default=1)
    ap.add_argument('-s', '--save', type=bool, default=True)
    ap.add_argument('-v', '--video', type=bool, default=False)
    ap.add_argument('-i', '--img_index', default=0)
    ap.add_argument('-sh', '--show', type=bool, default=False, help='Show average triangulation or not')
    args = ap.parse_args()

    main(args)

