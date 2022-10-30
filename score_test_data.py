import pandas as pd
import ast
import tqdm
from torchvision.io import read_image
from utils.model import ModelOcr

ADD_PATH = './Test/'


def main():
    test = pd.read_csv('./Test/test.csv')
    test['output'] = test.output.apply(lambda x: ast.literal_eval(x))
    image_path = []
    output = []
    new_output = []
    for i in tqdm.tqdm_notebook(test.iterrows(), total=test.shape[0]):
        row = i[1]
        image = row['image_path']
        out = row['output']
        for dict_of_vals in out[0]:
            output.append(out)
            image_path.append(image)
            new_output.append(dict_of_vals)
    test = pd.DataFrame({'old_image_path': image_path, 'output': output, 'new_output': new_output})
    test['image_path'] = test['old_image_path'].apply(lambda x: ADD_PATH + x)
    gt_boxes = []
    for i in tqdm.tqdm_notebook(test.iterrows(), total=test.shape[0]):
        row = i[1]
        image = row['image_path']
        box = row['new_output']
        img = read_image(image)
        bboxes = []
        labels = []
        scaler_x = img.shape[2]
        scaler_y = img.shape[1]
        if box['left'] <= 1.1 and box['top'] <= 1.1 and box['width'] <= 1.1 and box['height'] <= 1.1:
            bboxes.append([box['left'] * scaler_x, (box['top']) * scaler_y,
                           (box['left'] + box['width']) * scaler_x, scaler_y * (box['top'] + box['height'])])
        else:
            bboxes.append([box['left'], (box['top']),
                           (box['left'] + box['width']), (box['top'] + box['height'])])

        gt_boxes.append(bboxes)
    test['gt_bboxes'] = gt_boxes
    reader = ModelOcr(path_to_easy_osr=None)
    df = reader.predict_on_dataframe(test)
    df.pred_words = df.pred_words.apply(lambda x: ' '.join(x).replace('\n', ' '))
    for i in tqdm.tqdm_notebook(df.iterrows(), total=df.shape[0]):
        row = i[1]
        image = row['image_path']
        out = row['new_output']
        word = row['pred_words']
        out.update({'label': word})
    out_df = df.groupby(by=['old_image_path'])['new_output'].apply(list).reset_index()
    out_df['output'] = out_df.new_output.apply(lambda x: [x])
    out_df = out_df.rename(columns={'old_image_path': 'image_path'})[['image_path', 'output']]
    out_df.to_csv('Answers.csv', sep=',', index=False)

if __name__=="__main__":
    main()
