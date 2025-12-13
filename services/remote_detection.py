import requests
import io
import os

HF_MODEL = "facebook/detr-resnet-50"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
}

def detect_objects(image):
    """
    Returns a list of detections:
    [
      {
        "label": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2]
      }
    ]
    """

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        data=buf.getvalue(),
        timeout=60
    )

    response.raise_for_status()
    outputs = response.json()

    detections = []

    for obj in outputs:
        box = obj["box"]
        detections.append({
            "label": obj["label"],
            "confidence": obj["score"],
            "bbox": [
                box["xmin"],
                box["ymin"],
                box["xmax"],
                box["ymax"]
            ]
        })

    return detections
