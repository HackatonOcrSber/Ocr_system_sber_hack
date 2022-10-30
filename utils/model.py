import easyocr
from skimage import io
import os
from PIL import Image, JpegImagePlugin
import torch
import numpy as np
import cv2
from torchvision.io import read_image
import tqdm
from torchvision.utils import draw_bounding_boxes
import torchvision
from pathlib import Path


class ModelOcr:
    def __init__(self, path_to_easy_osr=None):
        self.path_to_easy_osr = path_to_easy_osr
        self.model = None
        self.initialize_model()
        self.images_idx = 0

    def initialize_model(self):
        if self.path_to_easy_osr is not None:
            self.model = easyocr.Reader(['ru', 'en'],
                                        model_storage_directory='/home/danil/.EasyOCR/model',
                                        user_network_directory='/home/danil/.EasyOCR/user_network',
                                        recog_network='custom_example')
        else:
            self.model = easyocr.Reader(['ru', 'en'])

    def clear_gpu_memory(self):
        if self.images_idx % 20 == 0:
            torch.cuda.empty_cache()
            self.initialize_model()

    def simple_predict_on_single_image(self, image_path):
        try:
            self.clear_gpu_memory()
            self.images_idx += 1
            predict = np.array(self.model.readtext(image_path))
            pred_words = list(predict[:, 1])
            pred_bboxes = list(predict[:, 0])
            pred_conf = list(predict[:, 2])
        except Exception as e:
            print(f"problem {e}")
            pred_words = ['']
            pred_bboxes = []
            pred_conf = []
        return (pred_bboxes, pred_words, pred_conf)

    def complicate_predict_on_single_image(self, image_path):
        image = cv2.imread(image_path)

        try:
            self.clear_gpu_memory()
            self.images_idx += 1
            img, img_cv_grey = self.reformat_input(image, False)
            horizontal_list, free_list = self.model.detect(img)
            horizontal_list, free_list = horizontal_list[0], free_list[0]
            predict = np.array(self.model.recognize(img_cv_grey, horizontal_list, free_list))
            if predict.shape != (0,):
                pred_words = list(predict[:, 1])
                pred_bboxes = list(predict[:, 0])
                pred_conf = list(predict[:, 2])
            else:
                pred_words = ['']
                pred_bboxes = []
                pred_conf = []
        except Exception as e:
            print(f"problem {e}")
            pred_words = ['']
            pred_bboxes = []
            pred_conf = []
        return (pred_bboxes, pred_words, pred_conf)

    def predict_on_folder(self, path_to_folder):
        images = os.listdir(path_to_folder)
        list_of_pred_bboxes = []
        list_of_pred_words = []
        list_of_pred_conf = []
        for image_path in images:
            pred_bboxes, pred_words, pred_conf = self.simple_predict_on_single_image(image_path)
            list_of_pred_bboxes += pred_bboxes
            list_of_pred_words += pred_words
            list_of_pred_conf += pred_conf
        return (list_of_pred_words, list_of_pred_bboxes, list_of_pred_conf)

    def loadImage(self, img_file):
        img = io.imread(img_file)  # RGB order
        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   img = img[:, :, :3]
        img = np.array(img)

        return img

    def reformat_input(self, image, mid_process=False):
        if type(image) == str:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = os.path.expanduser(image)
            img = self.loadImage(image)  # can accept URL
        elif type(image) == bytes:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif type(image) == np.ndarray:
            if len(image.shape) == 2:  # grayscale
                img_cv_grey = image
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                img_cv_grey = np.squeeze(image)
                img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
                img = image
                img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
                img = image[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif type(image) == JpegImagePlugin.JpegImageFile:
            image_array = np.array(image)
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('Invalid input type. Supporting format = string(file path or url), bytes, numpy array')
        if mid_process:
            def image_smoothening(img):
                ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
                ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                blur = cv2.GaussianBlur(th2, (1, 1), 0)
                ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return th3

            del img
            img = cv2.detailEnhance(img_cv_grey, sigma_s=50, sigma_r=0.15)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             41, 3)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            img = image_smoothening(img)
            img_cv_grey = cv2.bitwise_or(img, closing)
        return img, img_cv_grey

    def predict_with_gt(self, path_to_image, bboxes_gt):
        image = cv2.imread(path_to_image)
        pred_boxes = []
        pred_words = []
        for box in bboxes_gt:
            try:
                if box[1] < box[3] and box[0] < box[2] and box[2] < image.shape[1] and box[3] < image.shape[0]:
                    box[1] = max(0, box[1])
                    box[0] = max(0, box[0])
                    box = list(map(int, box))
                    self.clear_gpu_memory()
                    self.images_idx += 1
                    cropped_img = image[box[1]:box[3], box[0]:box[2]]
                    img, img_cv_grey = self.reformat_input(cropped_img, False)
                    horizontal_list, free_list = self.model.detect(img)
                    horizontal_list, free_list = horizontal_list[0], free_list[0]
                    predict = np.array(self.model.recognize(img_cv_grey, horizontal_list, free_list))

                    if predict.shape != (0,):
                        pred_words += list(predict[:, 1])
                        pred_boxes += list(predict[:, 0])
                    else:
                        pred_words += ['']
                        pred_boxes += []
            except Exception as e:
                print(f"problem {e}")
                pred_words += ['']
                pred_boxes += []
        return pred_words, pred_boxes

    def predict_on_dataframe(self, df, postprocessing=False, corrector=False):
        """
        Args:
            df: df[['image_path'],['gt_bboxes]]
            postprocessing : bool
            corrector : bool
        Returns:
            df: df[['image_path'],['gt_bboxes'], ['pred_boxes'], ['pred_words']
        """
        list_of_pred_boxes = []
        list_of_pred_words = []
        for idx, i in enumerate(tqdm.tqdm(df.iterrows(), total=df.shape[0])):
            row = i[1]
            image_path = row['image_path']
            bboxes_gt = row['gt_bboxes']
            pred_words, boxes = self.predict_with_gt(image_path, bboxes_gt)
            list_of_pred_boxes.append(boxes)
            list_of_pred_words.append(pred_words)
        df['pred_boxes'] = list_of_pred_boxes
        df['pred_words'] = list_of_pred_words
        df.reset_index(drop=True, inplace=True)
        if postprocessing:
            import post_process
            polygon_clusters = post_process.get_polygon_clusters(df, 0)
            global_family = post_process.get_clusters_family(df, polygon_clusters)
            global_output_text = post_process.get_output_text(df, global_family, polygon_clusters)
            df['pred_words'] = global_output_text
        if corrector:
            import corrector
            df['pred_words1'] = df['pred_words'].apply(lambda x: corrector.func(x))
        return df

    def get_4_coords(self, bbox):
        bbox = np.array(bbox)
        xmin = np.min(bbox[::, 0])
        ymin = np.min(bbox[::, 1])
        xmax = np.max(bbox[::, 0])
        ymax = np.max(bbox[::, 1])
        return xmin, ymin, xmax, ymax

    def proc_many_boxes(self, boxes):
        return [[*self.get_4_coords(i)] for i in boxes]

    def predict_and_save(self, path_to_orig_img):
        Path("./result").mkdir(parents=True, exist_ok=True)
        pred_bboxes, pred_words, pred_conf = self.complicate_predict_on_single_image(path_to_orig_img)
        img = read_image(path_to_orig_img)
        pred_bboxes = self.proc_many_boxes(pred_bboxes)

        bbox = torch.tensor(np.array(pred_bboxes), dtype=torch.int)

        img = draw_bounding_boxes(img, bbox, width=3, labels=[str(i) for i in range(bbox.shape[0])])
        img = torchvision.transforms.ToPILImage()(img)
        path_to_saved_img = os.path.join("./result", os.path.basename(path_to_orig_img))
        img.save(path_to_saved_img)
        return path_to_saved_img, (' '.join(pred_words).strip().replace('\n', ' '))