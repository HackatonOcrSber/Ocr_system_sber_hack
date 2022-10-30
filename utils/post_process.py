import cv2
import os
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd


def get_polygon_points(shape, x, y, w, h):
    upper_left_x = int(shape[1] * x)
    upper_left_y = int(shape[0] * y)
    down_right_x = int(upper_left_x + w * shape[1])
    down_right_y = int(upper_left_y + h * shape[0])

    return (upper_left_x, upper_left_y), (down_right_x, down_right_y)


def get_polygon_clusters(joined_df, threshold):
    polygon_clusters = dict()
    for i in tqdm_notebook(range(joined_df.shape[0])):
        im_path = joined_df["image_path"].iloc[i]
        img = cv2.imread(im_path)
        # if img is None:
        #   continue
        polygon_clusters_cur = []
        output = joined_df.loc[i, 'output'][0]
        for j in range(len(output)):
            x_true = output[j]['left']
            y_true = output[j]['top']
            width_true = output[j]['width']
            height_true = output[j]['height']

            polygon_clusters_cur.append(get_polygon_points(img.shape, x_true, y_true, width_true, height_true))

        preds = joined_df.loc[i, 'pred_boxes']
        img_polygon_pred = np.ones(shape=len(preds)) * -1
        for j in range(len(preds)):
            upper_left_pred = preds[j][0]
            down_right_pred = preds[j][2]
            for z in range(len(polygon_clusters_cur)):
                if upper_left_pred[0] >= polygon_clusters_cur[z][0][0] - threshold and \
                        upper_left_pred[1] >= polygon_clusters_cur[z][0][1] - threshold and \
                        down_right_pred[0] <= polygon_clusters_cur[z][1][0] + threshold and \
                        down_right_pred[1] <= polygon_clusters_cur[z][1][1] + threshold:
                    img_polygon_pred[j] = z

                    break
        polygon_clusters[im_path] = img_polygon_pred

    return polygon_clusters


def local2global(global_indexes, local_indexes, deleted_indexes):
    output = sorted(list(global_indexes - deleted_indexes))
    return list(np.array(output)[local_indexes])


def get_clusters_family(joined_df, polygon_clusters):
    global_family = []
    for img_num in tqdm_notebook(range(joined_df.shape[0])):

        pics = np.array(joined_df.loc[img_num, 'pred_boxes'])
        current_polygon_cluster = polygon_clusters[joined_df["image_path"].iloc[img_num]]

        big_family = []
        for cluster_num in sorted(list(set(current_polygon_cluster))):
            if cluster_num == -1:
                continue

            cluster_family = []
            nearest_family_y = []
            cur_pics = pics[current_polygon_cluster == cluster_num].copy()
            global_indexes = set(np.arange(cur_pics.shape[0]))
            global_deleted = set()
            while len(cur_pics) > 0:
                left_elem_index = np.argsort(cur_pics[:, 0, 0])[0]
                cur_family = [left_elem_index]
                for i in range(cur_pics.shape[0]):
                    if i == left_elem_index:
                        continue
                    else:
                        arr_y = np.array(
                            [cur_pics[cur_family[-1], 1, 1], cur_pics[cur_family[-1], 2, 1], cur_pics[i, 0, 1],
                             cur_pics[i, 3, 1]])
                        arr_y_ = np.array([1, 1, 2, 2])[np.argsort(arr_y)]
                        arr_y.sort()
                        intersection = arr_y[2] - arr_y[1]
                        if intersection >= 0.5 * (cur_pics[cur_family[-1], 2, 1] - cur_pics[cur_family[-1], 1, 1]) and \
                                arr_y_[0] != arr_y_[1]:
                            cur_family.append(i)
                cur_family_new = local2global(global_indexes, cur_family, global_deleted)
                cluster_family.append(cur_family_new)
                nearest_family_y.append(cur_pics[left_elem_index, 0, 1])
                cur_pics = np.delete(cur_pics, cur_family, axis=0)
                global_deleted |= set(cur_family_new)
            nearest_family_y = np.array(nearest_family_y)
            order = np.argsort(nearest_family_y)

            big_family.append((cluster_family, order))

        global_family.append(big_family)
    return global_family


def get_output_text(joined_df, global_family, polygon_cluster):
    global_output_text = []

    for img_num in tqdm_notebook(range(joined_df.shape[0])):
        pics = np.array(joined_df.loc[img_num, 'pred_boxes'])
        words_pred = joined_df.loc[img_num, 'pred_words']
        current_polygon_cluster = polygon_cluster[joined_df.loc[img_num, "image_path"]]
        current_output_text = []

        cluster_nums = set(current_polygon_cluster)
        if -1 in cluster_nums:
            cluster_nums.remove(-1)
        cluster_nums = sorted(cluster_nums)

        for j in range(len(global_family[img_num])):
            cur_family = global_family[img_num][j][0]
            order = global_family[img_num][j][1]
            polyg_ind = [i for i in range(len(current_polygon_cluster)) if
                         current_polygon_cluster[i] == cluster_nums[j]]
            cur_family_text = []
            for k in range(len(order)):
                init_ind = order[k]

                for z in range(len(cur_family[init_ind])):
                    cur_family_text.append(words_pred[polyg_ind[cur_family[init_ind][z]]])
            current_output_text.append(' '.join(cur_family_text))
        global_output_text.append(current_output_text)
    return global_output_text
