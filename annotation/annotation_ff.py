import cv2
import os
from tqdm import tqdm
import argparse
import glob
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import elasticdeform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter




def get_blur(fake, real):
    """
    ssim --> diff_face --> dilated --> hull --> eroded --> blur
    """

    d, a = compare_ssim(fake, real, full=True)
    diff = 1 - a

    diff_mask = (diff - diff.min()) / (diff.max() - diff.min())
    diff_mask = cv2.GaussianBlur(diff_mask, (5, 5), 0)
    factor = (diff_mask > 0.08) * 1
    diff_mask = fake * factor

    _, mask = cv2.threshold(np.uint8(diff_mask * 255), 0, 255, cv2.THRESH_BINARY)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel1, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours)):
        #print(contours[i])
        if contours[i].shape[0] > 5:
            if len(hull) == 0:
                hull = contours[i]
            else:
                hull = np.concatenate((hull, cv2.convexHull(contours[i], False)))

    sizey, sizex = fake.shape[0], fake.shape[1]
    dilated_hull = np.zeros((sizey, sizex, 3), dtype=np.uint8)
    dilated_hull = cv2.fillConvexPoly(dilated_hull, cv2.convexHull(hull, False), (255, 255, 255))

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(dilated_hull, kernel2, iterations=3)

    blur = cv2.GaussianBlur(eroded / 255, (7, 7), 0)
    _, blur = cv2.threshold(np.uint8(blur * 255), 0, 255, cv2.THRESH_BINARY)

    return np.uint8(diff_mask*255), np.uint8(blur)


def get_mask(file_real_dir ,file_fake_dir, file_mask_dir):
    file_fake_names = os.listdir(file_fake_dir)
    file_fake_names.sort()

    for name_fake in tqdm(file_fake_names):
        mask_path = os.path.join(file_mask_dir, name_fake)
        os.makedirs(mask_path, exist_ok=True)
        name_real = name_fake.split('_')[0]
        for img_name in os.listdir(os.path.join(file_fake_dir, name_fake)):

            img_name_real = os.path.join(file_real_dir, name_real, img_name)
            img_name_fake = os.path.join(file_fake_dir, name_fake, img_name)
            try:
                img_real = cv2.imread(img_name_real)
                img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY) / 255.
            except:
                print("Read File Error! file={}".format(img_name_real))
                continue

            try:
                img_fake = cv2.imread(img_name_fake)
                img_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY) / 255.
            except FileNotFoundError:
                continue
            
            try:
                diff_face,  blur = get_blur(img_fake, img_real)
                id = img_name.split('.')[0]
                cv2.imwrite(os.path.join(mask_path, id+'_'+'diff'+'.png'), diff_face)
                cv2.imwrite(os.path.join(mask_path, id+'_'+'mask'+'.png'), blur)
            except Exception as e:
                print(e)
                #os.remove(img_name_fake)
                print("Error File {}".format(img_name_fake))
                continue


if __name__ == '__main__':
    output_dir = "./annotation_mask/"
    realimage_path = "./FF++Raw/youtube/"

    for type in ['FaceSwap', 'Deepfakes', 'Face2Face', 'FaceShifter', 'NeuralTextures']:
        fakeimage_path = './FF++Raw/youtube/'+type+'/'
        mask_path = output_dir+type+'/'

        get_mask(realimage_path, fakeimage_path, mask_path)
    

        