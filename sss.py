

'''
from ultralytics import YOLO
from ultralytics.data import YOLODataset
import matplotlib.pyplot as plt
import cv2

# Load a YOLOv8 model from a .pt checkpoint
model = YOLO("yolov10s-seg.yaml")

# Define a small set of images
images = ['D:/auto_drive/dataset/complex_detection/Compete_COCO/images/train/Banseok01_Snow_00051254.png',
            'D:/auto_drive/dataset/complex_detection/Compete_COCO/images/train/Yongjun_Double_001235.png']

# Function to load and preprocess images
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Function to visualize augmentations
def visualize_augmentations(model, img_path):
    img = load_image(img_path)
    dataset = YOLODataset([img], augment=True, imgsz=640, stride=32)
    
    for augmented_img, _ in dataset:
        # Plot original and augmented images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[1].imshow(augmented_img.permute(1, 2, 0).numpy())
        ax[1].set_title('Augmented Image')
        plt.show()
        break  # Only visualize the first augmented image

# Apply data augmentation and visualize
for img_path in images:
    visualize_augmentations(model, img_path)
'''

from pathlib import Path

from PIL import Image

from ultralytics.utils import IterableSimpleNamespace
from ultralytics.data.dataset import YOLODataset

# create dataset config
default_cfg = IterableSimpleNamespace(task='segment', bgr=0., mode='test', model=None, data=None, epochs=100, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=True, optimizer='auto', verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split='val', save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, line_width=None, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format='torchscript', keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker='botsort.yaml')

my_dataset = YOLODataset(
    data={'names': ['0']},
    img_path="D:/auto_drive/dataset/complex_detection/Compete_COCO/images/train/Banseok01_Snow_00051254.png",
    imgsz=640,
    hyp=default_cfg,
    batch_size=1
)

augmented_image = next(iter(my_dataset))
Image.fromarray(augmented_image['img'].permute((1,2,0)).numpy())