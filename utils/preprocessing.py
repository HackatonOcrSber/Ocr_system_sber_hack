import tempfile
import cv2
import numpy as np
import argparse
import pandas as pd
from PIL import Image
import os
import tqdm
import shutil

# from scipy.ndimage import interpolation as inter

IMAGE_SIZE = 1000
BINARY_THREHOLD = 180


def process_image_for_ocr(file_path):
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    # im_new = cv2.dilate(im_new, np.ones((1,1),np.uint8), iterations=1)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path).convert("RGB")
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


# def find_score(arr, angle):
#     data = inter.rotate(arr, angle, reshape=False, order=0)
#     hist = np.sum(data, axis=1)
#     score = np.sum((hist[1:] - hist[:-1]) ** 2)
#     return hist, score

# def best_angle(img):
#     delta = 1
#     limit = 45
#     angles = np.arange(-limit, limit+delta, delta)
#     scores = []
#     for angle in angles:
#         hist, score = find_score(img, angle)
#         scores.append(score)
#     best_score = max(scores)
#     best_angle = angles[scores.index(best_score)]
#     print('Best angle: {}'.format(best_angle))

#     img = inter.rotate(img, best_angle, reshape=False, order=0)
#     #img = im.fromarray((255 * img).astype("uint8")).convert("RGB")
#     return img

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name)
    img = cv2.detailEnhance(img, sigma_s=50, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)

    # img = best_angle(or_image)
    return or_image


def main(args):
    data = pd.read_csv(os.path.join(args.data_root, "train.csv"))
    data = data.sort_values('image_path')

    # data["preprocessed_image_path"] = data["image_path"]
    os.mkdir(os.path.join(args.data_root, "Train_preprocessed"))
    os.mkdir(os.path.join(args.data_root, "Train_preprocessed", 'train'))

    for i in tqdm.tqdm(range(data.shape[0])):
        proc_image = process_image_for_ocr(os.path.join(args.data_root, data["image_path"].loc[i]))
        if (args.output_type == "RGB"):
            proc_image = cv2.cvtColor(proc_image, cv2.COLOR_GRAY2RGB)
        if args.negative:
            proc_image = 255 - proc_image
        cv2.imwrite(os.path.join(args.data_root, "Train_preprocessed", data["image_path"].loc[i]), proc_image)
        print(os.path.join(args.data_root, "Train_preprocessed", data["image_path"].loc[i]))
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--output-type', type=str, default='GREY')
    parser.add_argument('--negative', type=bool, default=False)
    args = parser.parse_args()

    main(args)
