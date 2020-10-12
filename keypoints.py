import matplotlib as plt
import argparse
import json
import os
from skimage.transform import rescale

from helpers import *

def main(args):
    # create file name for json 
    name = os.path.basename(args.img).split('.')[0]
    name2 = os.path.basename(args.img2).split('.')[0]
    file_name = f'out/{name}_{name2}.json'

    # read images as data
    img = read(args.img)
    img2 = read(args.img2)

    # rescale images to same height
    height = min(img.shape[1], img2.shape[1])
    img = rescale(img, height / img.shape[1], multichannel=True)
    img2 = rescale(img2, height / img2.shape[1], multichannel=True)

    # crop images to same width 
    width = min(img.shape[0], img2.shape[0])
    img = crop(img, (width, height))
    img2 = crop(img2, (width, height))

    # show images
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img)
    axarr[1].imshow(img2)

    if args.add:
        # read existing file to add data to
        with open(file_name, 'r') as f:
            all_points = json.load(f)
        
        for p in all_points['first']:
            axarr[0].scatter(x=p[0], y=p[1])
        
        for p in all_points['second']:
            axarr[1].scatter(x=p[0], y=p[1])

    else:
        all_points = {'first': [], 'second': []}
    
    # create keypoints
    for _ in range(args.num):
        points = plt.ginput(2)
        print(points)
        all_points['first'].append(points[0])
        all_points['second'].append(points[1])

        axarr[0].scatter(x=points[0][0], y=points[0][1])
        axarr[1].scatter(x=points[1][0], y=points[1][1])

    # save keypoints
    with open(file_name, 'w') as f:
        json.dump(all_points, f)
    
    # save cropped images
    save(img, 'out/' + name + '_crop.png')
    save(img2, 'out/' + name2 + '_crop.png')


    print(f"saved as {file_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img", default='in/divi.png')
    ap.add_argument("-j", "--img2", default='in/zap.png')
    ap.add_argument('-n', '--num', default=5, type=int)
    ap.add_argument("--add", type=bool, default=False, help='add points to existing json file')
    args = ap.parse_args()

    main(args)
