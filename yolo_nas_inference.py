import os
import torch

import requests
import torch
from PIL import Image

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)



# Set the device to GPU if available, otherwise fallback to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


best_model = models.get(
    model_name='yolo_nas_s',  # specify the model name here
    num_classes=2,
    checkpoint_path='yolo_nas\\average_model.pth'
).to(DEVICE)



best_model.predict_webcam( conf=0.25).show()
