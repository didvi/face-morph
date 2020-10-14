import numpy as np
import argparse
import glob
import matplotlib as plt
import skimage.transform

from morph import *
from helpers import *
from mean import *

def main(args):
    file_name = 'out/divi_large_to_danes.json'

    # read keypoints 
    with open(file_name, 'r') as f:
        all_points = json.load(f)
    
    points = all_points['first']
    points = np.array(points)

    # get file names
    files = get_by_face_type(type=args.type)
    img_paths = get_by_face_type(type=args.type, file_ext='.jpg')

    # read data 
    img = read(args.img)
    height, width = img.shape[:2]

    # calculate average points
    avg_x, avg_y = calc_avg_points(files)
    avg_points = format_points(avg_x, avg_y, width, height)

    # create average image
    avg_morphed_img = avg_morph(img_paths, files, avg_points)
    
    # morph my face to average image
    divi_to_mean = morph_to_points(img, points, avg_points)
    show(divi_to_mean)

    # morph average face to my points
    mean_to_divi = morph_to_points(avg_morphed_img, avg_points, points)
    show(mean_to_divi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=int, default=1)
    ap.add_argument('-i', '--img', type=str, default='out/divi_large_crop.png')
    args = ap.parse_args()

    main(args)

