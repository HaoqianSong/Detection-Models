from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import time
import argparse
import yaml
import os
from tqdm.auto import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, default="data.yaml", #required=True,
                help="path to data.yaml")
ap.add_argument("-m", "--model", type=str, default="yolo_nas_l", #required=True,
                help="Model type (eg: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, default="D:/DetectionAlgorithm/DataSet/YOLO/YOLONAS_Loutput/train/2GPU/ckpt_best.pth", #required=True,
                help="path to trained model weight")
ap.add_argument("-s", "--source", type=str, default="Data/test/images/",  #required=True,
                help="video path/cam-id/RTSP")
ap.add_argument("-o", "--outpath", type=str, default="D:/DetectionAlgorithm/DataSet/YOLO/YOLONAS_Loutput/predect1/",
                    help="Predict output dir name")
ap.add_argument("-c", "--conf", type=float, default=0.45,
                help="model prediction confidence (0<conf<1)")
ap.add_argument("--save", action='store_true',
                help="Save video")
ap.add_argument("--hide", action='store_false', default=False,
                help="to hide inference window")

args = vars(ap.parse_args())
yaml_params = yaml.safe_load(open(args['data'], 'r'))

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_bbox(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = next(model.predict(img_rgb, conf=args['conf'])._images_prediction_lst)
    class_names = preds.class_names
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        plot_one_box(box[:4], img, label=f'{class_names[cs]} {cnf:.3}', color=colors[cs])
    return labels, class_names, bboxes, confs


# Load YOLO-NAS Model
model = models.get(
    args['model'],
    num_classes=len(yaml_params['names']), 
    checkpoint_path=args["weight"]
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
print('Class Names: ', yaml_params['names'])
colors = [[170, 0, 170] for _ in yaml_params['names']]     #[random.randint(0, 255) for _ in range(3)]

# Inference Image
all_images = os.listdir(args['source'])
#if args['source'].endswith('.jpg') or args['source'].endswith('.jpeg') or args['source'].endswith('.png'):
#tot = 0.0 #%%
for image in tqdm(all_images, total=len(all_images)):
    image_path = os.path.join(args['source'], image)
    img = cv2.imread(image_path)
    labels, class_names, bboxes0, confs0 = get_bbox(img)
    #start_time = time.time()   #%%
    #out = model.predict(img, conf=args['conf'])
    #prediction_time = time.time() - start_time   #%%
    #tot += prediction_time  #%%
    #print(f"Each prediction time: {prediction_time * 1E3:.2f} ms")  #%%
    print(labels)
    if args['hide'] is False and len(labels)>0:
        pre_list = [class_names[x] for x in labels]
        count_pred = {i:pre_list.count(i) for i in pre_list}
        print(f'Prediction: {count_pred}')

    # save Image
    if args['save'] or args['hide'] is False:
        os.makedirs(os.path.join(args['outpath'], '1test45_2GPU'), exist_ok=True)
        os.makedirs(os.path.join(args['outpath'], '1test45_2GPU', 'labels'), exist_ok=True)
        path_save = os.path.join(args['outpath'], '1test45_2GPU', os.path.split(image_path)[1])  # args['source']
        path_txt = os.path.join(args['outpath'], '1test45_2GPU', 'labels', os.path.split(image_path)[1])
        file_name, file_extension = os.path.splitext(path_txt)
        path_txt_save = file_name + '.txt'
        cv2.imwrite(path_save, img)
        zero_column = np.zeros((bboxes0.shape[0], 1))
        boxes_array = np.concatenate((zero_column, confs0.reshape(-1,1), bboxes0), axis=1)
        boxes_array[boxes_array<0] = 0
        boxes_array1 = np.array2string(boxes_array, precision=5, suppress_small=True)#
        boxes_str=boxes_array1.replace('e+0', '').replace('e-', 'e-')
        with open(path_txt_save, 'w') as file:
            file.write(boxes_str)
        print(f"[INFO] Saved Image: {path_save}")

    
    # Hide video
    if args['hide']:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
#print(f"Total prediction time: {tot * 1E3:.2f} ms")  #%%

'''
# Reading Video/Cam/RTSP
else:
    video_path = args['source']
    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

    if args['hide'] is False:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

    # Get the width and height of the video - SAVE VIDEO.
    if args['save'] or args['hide'] is False:
        original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        if not str(video_path).isnumeric():
            path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
        else:
            c = 0
            while True:
                if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
                    path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
                    break
                else:
                    c += 1
        out_vid = cv2.VideoWriter(path_save, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (original_video_width, original_video_height))

    p_time = 0
    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] Failed to read...')
            break
        
        labels, class_names = get_bbox(img)
        if args['hide'] is False and len(labels)>0:
            frame_count += 1
            pre_list = [class_names[x] for x in labels]
            count_pred = {i:pre_list.count(i) for i in pre_list}
            print(f'Frames Completed: {frame_count}/{length} Prediction: {count_pred}')
            
        # FPS
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        cv2.putText(
            img, f'FPS: {fps:.3}', (50, 60),
            cv2.FONT_HERSHEY_PLAIN, 2, 
            (0, 255, 0), 2
        )

        # Write Video
        if args['save'] or args['hide'] is False:
            out_vid.write(img)

        # Hide video
        if args['hide']:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args['save'] or args['hide'] is False:
        out_vid.release()
        print(f"[INFO] Outout Video Saved in {path_save}")
    if args['hide']:
        cv2.destroyAllWindows()
'''