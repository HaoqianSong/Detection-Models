import numpy as np
import cv2
import torch
import os

label_path = './test_label'
image_path = './test'


# 坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywhn2xyxy(x, w=800, h=800, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


folder = os.path.exists('GT')
if not folder:
    os.makedirs('GT')

folderlist = os.listdir(label_path)
for i in folderlist:
    label_path_new = os.path.join(label_path, i)
    with open(label_path_new, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # predict_label

    read_label = label_path_new.replace(".txt", ".png")
    read_label_path = read_label.replace('test_label', 'test')
    print(read_label_path)
    img = cv2.imread(str(read_label_path))
    h, w = img.shape[:2]
    lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
    for _, x in enumerate(lb):
        class_label = int(x[0])  # class
        cv2.rectangle(img, (round(x[1]), round(x[2])), (round(x[3]), round(x[4])), (0, 255, 0))
        with open('GT/' + i, 'a') as fw:
            fw.write(str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + ' ' + str(
                x[4]) + '\n')

