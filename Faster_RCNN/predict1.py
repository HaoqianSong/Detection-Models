import os
import time
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def detect(Img_Path, Model, save_name, imwrite=True):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=2)  #21

    # load train weights
    weights_path = Model   #./save_weights/model.pth
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open(Img_Path)   #./test.jpg

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        EachT = (t_end - t_start)   #*1000
        #print("inference+NMS time: {}ms".format((t_end - t_start)*1000))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        #print(predict_boxes)
        plot_img, boxes, classes, scores = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        Classes=classes-1
        boxes_array = np.concatenate((Classes.reshape(-1, 1), scores.reshape(-1, 1), boxes), axis=1)
        boxes_array[boxes_array<0] = 0
        boxes_array1 = np.array2string(boxes_array, precision=5, suppress_small=True)#
        boxes_str=boxes_array1.replace('e+0', '').replace('e-', 'e-')

        if imwrite:
            if not os.path.exists("D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/detect/2GPU983res50/0.5"):  #########
                os.makedirs('D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/detect/2GPU983res50/0.5')     ###########
            plot_img.save('D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/detect/2GPU983res50/0.5/{}'.format(save_name))  ##########
        with open('D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/detect/2GPU983res50/0.5/label/{}'.format(save_name[:-4]) +'.txt', 'w') as file:
            file.write(boxes_str)
        return EachT

if __name__ == '__main__':
        # <--修改为待预测图片所在文件夹路径
    Test_list_path = "D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/test/"
    files = os.listdir(Test_list_path)
    Weights_Path = "D:/DetectionAlgorithm/DataSet/R-CNN/faster_R-CNN/output/train983/2GPU983res50/model_299.pth"

    cn = 0
    TotalT = 0
    for file in files:
        img_path = os.path.join(Test_list_path, file)
        print(file)
        EachT = detect(img_path, Weights_Path, file, imwrite=True)
        print("{} [INFO] {} time: {} done!!!".format(cn, file, EachT))

        cn += 1
        TotalT += EachT
        waste_avg = TotalT / cn
    print(f"Total prediction time: {TotalT * 1E3:.2f} ms")
    print(f"Each prediction time: {waste_avg * 1E3:.2f} ms")
    print(f"FPS: {1000/(waste_avg * 1E3):.2f} ")
