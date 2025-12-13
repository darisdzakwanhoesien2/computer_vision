from models.resnet_model import classify_resnet
from models.efficientnet_model import classify_efficientnet
from utils.imagenet_labels import IMAGENET_LABELS

def run_classification(image, model_name, topk=5):
    if model_name == "resnet":
        idxs, scores = classify_resnet(image, topk)
    elif model_name == "efficientnet":
        idxs, scores = classify_efficientnet(image, topk)
    else:
        raise ValueError("Unknown model")

    results = [
        {
            "label": IMAGENET_LABELS[i],
            "confidence": float(s)
        }
        for i, s in zip(idxs, scores)
    ]

    return results

# from models.yolo_model import yolo_inference
# from models.frcnn_model import frcnn_inference

# def run_inference(image, model_name, confidence):
#     if model_name == "yolo":
#         return yolo_inference(image, confidence)
#     elif model_name == "faster_rcnn":
#         return frcnn_inference(image, confidence)
#     else:
#         raise ValueError("Unknown model selected")
