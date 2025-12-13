from models.resnet_model import classify_resnet
from models.efficientnet_model import classify_efficientnet

def run_classification(image, model_name="resnet", topk=5):
    """
    Runs image classification and returns a list of dicts:
    [
        {"label": str, "confidence": float},
        ...
    ]
    """

    if model_name == "resnet":
        results = classify_resnet(image, topk)

    elif model_name == "efficientnet":
        results = classify_efficientnet(image, topk)

    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    # 🔒 Contract enforcement (optional but safe)
    if not isinstance(results, list):
        raise TypeError("Model did not return a list")

    return results

from services.remote_detection import detect_objects

def run_detection(image):
    """
    Returns:
    [
      {
        "label": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2]
      },
      ...
    ]
    """
    try:
        return detect_objects(image)
    except RuntimeError as e:
        return {"error": str(e)}


# from models.resnet_model import classify_resnet
# from models.efficientnet_model import classify_efficientnet

# def run_classification(image, model_name="resnet", topk=5):
#     if model_name == "resnet":
#         return classify_resnet(image, topk)

#     elif model_name == "efficientnet":
#         return classify_efficientnet(image, topk)

#     else:
#         raise ValueError("Unknown classification model")

# if 'label' in r and 'confidence' in r:
#     try:
#         confidence = float(r['confidence'])
#         st.write(f"**{r['label']}** — {confidence:.3f}")
#     except (ValueError, TypeError):
#         st.error("Confidence value is not a valid number.")
# else:
#     st.error("Invalid classification result format.")


# from models.resnet_model import classify_resnet
# from models.efficientnet_model import classify_efficientnet
# from utils.imagenet_labels import IMAGENET_LABELS

# def run_classification(image, model_name, topk=5):
#     if model_name == "resnet":
#         idxs, scores = classify_resnet(image, topk)
#     elif model_name == "efficientnet":
#         idxs, scores = classify_efficientnet(image, topk)
#     else:
#         raise ValueError("Unknown model")

#     results = [
#         {
#             "label": IMAGENET_LABELS[i],
#             "confidence": float(s)
#         }
#         for i, s in zip(idxs, scores)
#     ]

#     return results

# from models.yolo_model import yolo_inference
# from models.frcnn_model import frcnn_inference

# def run_inference(image, model_name, confidence):
#     if model_name == "yolo":
#         return yolo_inference(image, confidence)
#     elif model_name == "faster_rcnn":
#         return frcnn_inference(image, confidence)
#     else:
#         raise ValueError("Unknown model selected")