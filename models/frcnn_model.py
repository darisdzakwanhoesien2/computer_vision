import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
from config.labels import COCO_CLASSES

_model = None

def load_frcnn():
    global _model
    if _model is None:
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        _model.eval()
    return _model

def frcnn_inference(image, confidence):
    model = load_frcnn()
    tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)[0]

    annotated = np.array(image)
    classes = []

    for box, label, score in zip(
        output["boxes"], output["labels"], output["scores"]
    ):
        if score >= confidence:
            cls = COCO_CLASSES[label]
            classes.append(cls)

            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{cls} {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    return classes, annotated
