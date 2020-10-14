import matplotlib as plt
import argparse
import json
import os
from skimage.transform import rescale

from morph import *
from helpers import *
from mean import *

def main(args):
    # create file name for json 
    name = os.path.basename(args.img).split('.')[0]
    file_name = f'out/{name}_to_danes.json'

    # get file names
    files = get_by_face_type(type=args.type)
    img_paths = get_by_face_type(type=args.type, file_ext='.jpg')
    

    # read reference image data 
    reference = read(img_paths[args.img_index])
    height, width = reference.shape[:2]
    x, y = read_asf(files[args.img_index])
    ref_points = format_points(x, y, width, height)

    # read image as data
    img = read(args.img)
    print(img.shape, reference.shape)
    
    # rescale images to same width
    img = rescale(img, height / img.shape[0], multichannel=True)
    print(img.shape, reference.shape)

    # crop image to same height 
    img = crop(img, (height, width))
    plt.rcParams["figure.figsize"] = (20,10)
    # create keypoints
    corresponding_points = []
    for point in ref_points:
        # Plot the point and image in a figure
        figure = plt.figure()
        figure.add_subplot(1, 2, 1)
        plt.imshow(reference)
        plt.scatter(point[0], point[1])

        # Plot the new image
        figure.add_subplot(1, 2, 2)
    
        plt.imshow(img)

        point = plt.ginput(1)
        corresponding_points += [point[0]]

        plt.close(figure)

    # save keypoints
    print(corresponding_points)
    with open(file_name, 'w') as f:
        json.dump({'first':corresponding_points}, f)

    print(f"saved as {file_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=int, default=1)
    ap.add_argument("-i", "--img", default='in/divi_large.png')
    ap.add_argument("-j", "--img_index", default=0)
    args = ap.parse_args()

    main(args)
