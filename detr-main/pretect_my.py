#https://blog.csdn.net/qq_45836365/article/details/128252220
#https://blog.csdn.net/m0_46412065/article/details/128538040
import glob
import math
import argparse
import numpy as np
from models.detr import DETR
from models.backbone import Backbone, build_backbone
from models.transformer import build_transformer
from PIL import Image
import cv2
import requests
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torchvision.models as models
torch.set_grad_enabled(False)
import os
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,                        help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,                        help="Path to the pretrained model. If set, only the mask head will be trained")    # * Backbone
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    # * Segmentation
    parser.add_argument('--masks', action='store_true',                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',                        help="Disables auxiliary decoding losses (loss at each layer)")    # * Matcher
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,                        help="giou box coefficient in the matching cost")    # * Loss coefficients
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,                        help="Relative classification weight of the no-object class")
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='D:/DetectionAlgorithm/DETR/1End-to-End Object Detection with Transformers/detr-main/data/coco', type=str)  ############
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='output',  help='path where to save, empty for no saving')   #########
    parser.add_argument('--device', default='cuda',    help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='detr_r50_91.pth', help='resume from checkpoint')   #####
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
'''
CLASSES = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
         "hair drier", "toothbrush"]
'''
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
transform_input = T.Compose([T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
    (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

'''
def plot_results(pil_img, prob, boxes, save_path):
    lw= max(round(sum(pil_img.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
           c1=p.argmax()
           text=f'{CLASSES[c1]}:{p[c1]:0.2f}'
           cv2.rectangle(pil_img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), colors(c1,True), thickness=lw,lineType=cv2.LINE_AA)
           if text:
           	tf=max(lw-1,1)
                w,h=cv2.getTextSize(text,0,fontScale=lw/3,thickness=tf)[0]
                cv2.rectangle(pil_img,(int(xmin),int(ymin)), (int(xmin)+w,int(ymin)-h-3),colors(c1,True),-1,cv2.LINE_AA)
                cv2.putText(pil_img, text, (int(xmin), int(ymin) - 2), 0, lw / 3, (255,255,255), thickness=tf,                        lineType=cv2.LINE_AA)
    Image.fromarray(ori_img).save(save_path)
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
backbone=build_backbone(args)
transform=build_transformer(args)
model=DETR(backbone=backbone,transformer=transform,num_classes=58,num_queries=100)
model_path='G:/DetectionAlgorithm/DETR/1End-to-End Object Detection with Transformers/detr-main/output/checkpoint0099.pth'   #/home/nianliu/wangxx/detr/cdnet_weights/checkpoint0179.pth保存的预训练好的模型pth文件，用于验证
model_data=torch.load(model_path)['model']
model=torch.load(model_path)model.load_state_dict(model_data)
model.eval();

paths = os.listdir('G:/DetectionAlgorithm/DETR/1End-to-End Object Detection with Transformers/detr-main/data/pretect')  #待验证的图片路径/home/nianliu/wangxx/detr/images
for path in paths:    # 问题1：无法读取png图像
    if os.path.splitext(path)[1] == ".png":    # 问题1解1：用imread读取png
       im = cv2.imread(path)
       im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    else:
        im = Image.open('G:/DetectionAlgorithm/DETR/1End-to-End Object Detection with Transformers/detr-main/data/pretect'+'/'+path) #/home/nianliu/wangxx/detr/images
    # mean-std normalize the input image (batch-size: 1)
        img = transform_input(im).unsqueeze(0)
    # propagate through the model
    outputs = model(img)
    # keep only predictions with 0.9+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    #保存验证结果地址
    img_save_path = 'G:/DetectionAlgorithm/DETR/1End-to-End Object Detection with Transformers/detr-main/data/pretect_result' + os.path.splitext(os.path.split(path)[1])[0] + '.jpg'  #/home/nianliu/wangxx/detr/infer_results/
    ori_img=np.array(im)
    plot_results(ori_img, probas[keep], bboxes_scaled, img_save_path)

'''

def plot_results(pil_img, prob, boxes, img_save_path):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}:      {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=9,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.savefig(img_save_path)
    plt.axis('off')
    plt.show()


def main(num_classes, chenkpoint_path, img_path, img_save_path, num_queries=100):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    backbone = build_backbone(args)
    transform = build_transformer(args)
    model = DETR(backbone=backbone, transformer=transform, num_classes=num_classes, num_queries=100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model_path = chenkpoint_path
    model_data = torch.load(model_path)['model']
    model.load_state_dict(model_data)
    model.eval()
    path = img_path
    im = cv2.imread(path)
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    img = transform_input(im).unsqueeze(0)
    outputs = model(img.to(device))
    probs = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # 可修改阈值,只输出概率大于0.7的物体
    keep = probs.max(-1).values > 0.7
    # print(probs[keep])
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    ori_img = np.array(im)
    plot_results(ori_img, probs[keep], bboxes_scaled, img_save_path)


if __name__ == "__main__":
    CLASSES = ["person"]
    main(num_classes=1, chenkpoint_path="D:/DetectionAlgorithm/DataSet/DETR/DETRoutput/checkpoint0499.pth",
         img_path="D:/DetectionAlgorithm/DataSet/DETR/DETRinput/test2017/",
         img_save_path="D:/DetectionAlgorithm/DataSet/DETR/DETRoutput/pretect")
