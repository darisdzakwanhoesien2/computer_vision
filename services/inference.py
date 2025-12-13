from models.yolo_model import yolo_inference
from models.frcnn_model import frcnn_inference

def run_inference(image, model_name, confidence):
    if model_name == "yolo":
        return yolo_inference(image, confidence)
    elif model_name == "faster_rcnn":
        return frcnn_inference(image, confidence)
    else:
        raise ValueError("Unknown model selected")
