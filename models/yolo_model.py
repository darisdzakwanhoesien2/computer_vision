from ultralytics import YOLO
import cv2

_model = None

def load_yolo():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def yolo_inference(image, confidence):
    model = load_yolo()
    results = model(image)[0]

    classes = []
    for box in results.boxes:
        if float(box.conf) >= confidence:
            classes.append(model.names[int(box.cls)])

    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return classes, annotated
