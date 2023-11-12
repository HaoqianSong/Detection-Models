
import os, sys
import torch, json
import numpy as np
import time
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

from PIL import Image
import datasets.transforms as T

'''https://blog.csdn.net/pengxiang1998/article/details/131445470
#%%%%%%%%%%%% load data 1方式
args.dataset_file = 'coco'
args.coco_path = "D:/DetectionAlgorithm/DataSet/DETR/DETRinput" # /comp_robot/cv_public_dataset/COCO2017/    the path of coco
args.fix_size = False
dataset_val = build_dataset(image_set='test', args=args)   #val
image, targets = dataset_val[0]
'''

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect(model,image1,id2name,Save_path,name):
    with torch.no_grad():
        image, _ = transform(image1, None)
        start = time.time()
        # %%%%%%%%%%%%%%predict images
        output = model.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        ###%%%%%%%%%%%%%%%可视化预测
        thershold = 0.45 # set a thershold
        vslzr = COCOVisualizer()
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > thershold
        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'scores': scores[select_mask],
            #'size': targets['size'],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'box_label': box_label
        }
        end = time.time()
        EachT = end - start
        vslzr.visualize(image, pred_dict, name, savedir=Save_path, dpi=100)
        BBoxes = pred_dict['boxes'].tolist()
        SScores = pred_dict['scores'].tolist()
        Boxes = np.array(BBoxes)
        Scores = (np.array(SScores)).reshape(-1, 1)
        zero_column = np.zeros((Boxes.shape[0], 1))
        boxes_array = np.concatenate((zero_column, Scores, Boxes), axis=1)#
        boxes_array[boxes_array<0] = 0
        boxes_array1 = np.array2string(boxes_array, precision=5, suppress_small=True)#
        boxes_str=boxes_array1.replace('e+0', '').replace('e-', 'e-')
        os.makedirs(os.path.dirname(Save_path+'/label/{}'), exist_ok=True)
        with open(Save_path+'/label/{}'.format(name[:-4]) +'.txt', 'w') as file:
            file.write(boxes_str)
    return EachT

if __name__ == "__main__":
    model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
    model_checkpoint_path = "D:/DetectionAlgorithm/DataSet/DETR/DINOoutput/train4-500/checkpoint_best_regular.pth" #checkpoint_best_regular.pth checkpoint0023_4scale.pth    change the path of the model checkpoint
    # See our Model Zoo section in README.md for more details about our pretrained models.
    Save_path = "D:/DetectionAlgorithm/DataSet/DETR/DINOoutput/detect/4-500/50"
    list_path = "D:/DetectionAlgorithm/DataSet/DETR/DETRinput/test2017" #
    files = os.listdir(list_path)

    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path)  #, map_location='cpu'
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    # load coco names
    with open('util/coco_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k):v for k,v in id2name.items()}

    cn = 0
    TotalT = 0
    for file in files:
        img_path = os.path.join(list_path, file)
        im = Image.open(img_path)
        EachT = detect(model,im,id2name,Save_path,file)
        print("{} [INFO] {} time: {} done!!!".format(cn, file, EachT))#

        cn += 1
        TotalT += EachT
        waste_avg = TotalT / cn
    print(f"Total prediction time: {TotalT * 1E3:.2f} ms")
    print(f"Each prediction time: {waste_avg * 1E3:.2f} ms")
    print(f"FPS: {1000/(waste_avg * 1E3):.2f} ")
