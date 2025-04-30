import numpy as np
import torch
from ultralytics import YOLO

# YOLOv11
class YOLOv11Detector:
    def __init__(self, ckpt='/home/y_li/workspace3/ultralytics/yolov11-finetune/train/weights/best.pt', conf=0.2, device='cuda'):
        self.model = YOLO(ckpt)
        self.conf = conf
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, im_bgr):
        res = self.model.predict(im_bgr, verbose=False, conf=self.conf)[0]
        if not res.boxes:
            return np.empty((0, 6), dtype=np.float32)
        
        xyxy = res.boxes.xyxy.cpu().numpy()
        score = res.boxes.conf.cpu().numpy()[:, None]
        cls = res.boxes.cls.cpu().numpy()[:, None]
        return np.hstack([xyxy, score]).astype(np.float32)
    
   


