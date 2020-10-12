import argparse
import json
import matplotlib
import skimage.draw
import imageio 
import cv2

from scipy.spatial import Delaunay

from helpers import *

def compute_transform(tri1, tri2):
    """Computes affine transform from triangle one to triangle two"""
    tri2 = np.hstack((tri2, np.ones((3, 1))))
    return np.vstack((tri1.T @ np.linalg.inv(tri2.T), np.array([0, 0, 1])))

def morph(img1, img2, keypoints, alpha=0.5):
    """Computes mid-way face of two images according to keypoints

    Args:
        img1 (np.ndarray): first image
        img2 (np.ndarray): second image
        keypoints (dict): dictionary of points with keys "first" and "second". Use keypoints.py to generate
    """
    if img1.shape[:1] != img2.shape[:1]:
        print("images not same shape")
        return

    # compute triangulation
    key1 = keypoints['first']
    key2 = keypoints['second']

    averaged_points = [
        [points[0][0] * alpha + points[1][0] * (1 - alpha),
         points[0][1] * alpha + points[1][1] * (1 - alpha)]
        for points in zip(key1, key2)
    ]

    triangulation = Delaunay(averaged_points)

    morphed_img = np.zeros(img1.shape)

    # compute affine transformations
    for t in triangulation.simplices:
        first_tri = np.array([key1[i] for i in t])
        second_tri = np.array([key2[i] for i in t])
        avg_tri = np.array([averaged_points[i] for i in t])

        # compute affine transformation matrix
        transform1 = compute_transform(first_tri, avg_tri)
        transform2 = compute_transform(second_tri, avg_tri)

        # compute indices needed
        rr, cc = sk.draw.polygon(avg_tri.T[1], avg_tri.T[0])

        # compute indices needed
        first_indices = np.column_stack((cc, rr, np.ones(rr.shape[0])))
        first_indices = transform1 @ first_indices.T

        second_indices = np.column_stack((cc, rr, np.ones(rr.shape[0])))
        second_indices = transform2 @ second_indices.T
    
        # shit so it works
        def make_indexable(arr):
            arr[0] = (np.clip(arr[0], 0, img1.shape[0] - 1))
            arr[1] = (np.clip(arr[1], 0, img1.shape[1] - 1))
            return arr.astype(int)
        
        first_indices = make_indexable(first_indices)
        second_indices = make_indexable(second_indices)
        
        # find pixels and average
        morph1 = alpha * img1[first_indices[1], first_indices[0]] 
        morph2 = (1 - alpha) * img2[second_indices[1], second_indices[0]]
        morphed_img[rr, cc] = morph1 + morph2



    # debug code TODO remove
    # key1 = np.array(key1)
    # key2 = np.array(key2)

    # averaged_points = np.mean([keypoints['first'], keypoints['second']], axis=0)
    # triangulation = Delaunay(averaged_points)
    # plt.triplot(matplotlib.tri.Triangulation(key2.T[0], key2.T[1]))
    # plt.triplot(matplotlib.tri.Triangulation(averaged_points.T[0], averaged_points.T[1]))
    # plt.imshow(img2)
    # plt.show()

    # plt.triplot(matplotlib.tri.Triangulation(key1.T[0], key1.T[1]))
    # plt.triplot(matplotlib.tri.Triangulation(averaged_points.T[0], averaged_points.T[1]))
    # plt.imshow(img1)
    # plt.show()
    return toInt(morphed_img)

def create_video(img1, img2, keypoints, video_name, frame_count=60):

    writer = imageio.get_writer(f'{video_name}.mp4', fps=15)

    alphas = np.linspace(0, 1, frame_count)
    print(alphas)
    for alpha in alphas:
        frame = morph(img1, img2, keypoints, alpha=alpha)
        writer.append_data(frame)
        print(f"Alpha: {alpha}")
        # maybe opencv is good for something
        cv2.imshow('Frame', frame) 
        # Press S on keyboard to stop the process 
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
    
    writer.append_data(toInt(img1))
    writer.close()
    
    print("The video was successfully saved") 

def main(args):
    if not args.json:
        # assume json name if not found
        name = os.path.basename(args.img).split('.')[0][:-5]
        name2 = os.path.basename(args.img2).split('.')[0][:-5]
        file_name = f'out/{name}_{name2}.json'
    else:
        file_name = args.json

    # read keypoints 
    with open(file_name, 'r') as f:
        all_points = json.load(f)

    # read images
    img = read(args.img)
    img2 = read(args.img2)

    img = img[:, :, :3]
    img2 = img2[:, :, :3]

    # create video
    if args.video:
        create_video(img, img2, all_points, f'out/morph_{name}_{name2}')
        return
    else:
        # morph image
        morphed_img = morph(img, img2, all_points)
        
        # show and save
        show(morphed_img)
        if args.save:
            save(morphed_img, f'morph_{name}_{name2}.png')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img", default='out/divi_crop.png')
    ap.add_argument("-j", "--img2", default='out/zap_crop.png')
    ap.add_argument('--json', help='json file containing keypoints')
    ap.add_argument('-s', '--save', type=bool, default=False)
    ap.add_argument("--add", type=bool, default=False, help='add points to existing json file')
    ap.add_argument('-v', '--video', type=bool, default=False)
    args = ap.parse_args()

    main(args)
