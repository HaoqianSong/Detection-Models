import cv2
from PIL import Image
import numpy as np
import os
import time
#https://blog.csdn.net/weixin_44635198/article/details/128429470
import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model

torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))

# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL =['person']
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("D:/DetectionAlgorithm/DataSet/DETR/DDETRoutput/pretect/1"):
            os.makedirs('D:/DetectionAlgorithm/DataSet/DETR/DDETRoutput/pretect/1')
        cv2.imwrite('D:/DetectionAlgorithm/DataSet/DETR/DDETRoutput/pretect/1/{}'.format(save_name), opencvImage)

# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

def load_model(model_path , args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load("D:/DetectionAlgorithm/DataSet/DETR/DDETRoutput/checkpoint0499.pth") # <------model_path-----修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model

# 图像的推断
def detect(im, model, transform, prob_threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)


    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, end - start


if __name__ == "__main__":

    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('D:/DetectionAlgorithm/DataSet/DETR/DDETRoutput/checkpoint0499.pth',main_args) # <---DETR/r50_deformable_detr-checkpoint.pth---修改为自己加载模型的路径

    files = os.listdir("D:/DetectionAlgorithm/DataSet/DETR/DETRinput/test2017/") # <---DDETR/data/coco/mytestdata/---修改为待预测图片所在文件夹路径

    cn = 0
    waste=0
    for file in files:
        img_path = os.path.join("D:/DetectionAlgorithm/DataSet/DETR/DETRinput/test2017/", file) # <---DDETR/data/coco/mytestdata/---修改为待预测图片所在文件夹路径
        im = Image.open(img_path)

        scores, boxes, waste_time = detect(im, dfdetr, transform)
        plot_result(im, scores, boxes, save_name=file, imshow=False, imwrite=True)
        print("{} [INFO] {} time: {} done!!!".format(cn,file, waste_time))

        cn+=1
        waste+=waste_time
        waste_avg = waste/cn
        print(waste_avg)