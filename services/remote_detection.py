import requests
import io
import os
import time

HF_MODEL = "facebook/detr-resnet-50"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
} if HF_TOKEN else {}

def detect_objects(image, max_retries=3, wait_seconds=5):
    """
    Calls Hugging Face Inference API safely.
    NEVER raises requests.HTTPError.
    Always returns either:
      - list of detections
      - dict {"error": "..."}
    """

    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set in Streamlit secrets."}

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    for attempt in range(max_retries):

        try:
            response = requests.post(
                HF_API_URL,
                headers=HEADERS,
                data=buf.getvalue(),
                timeout=60
            )
        except requests.RequestException as e:
            return {"error": f"Network error: {e}"}

        # ----------------------------
        # Hugging Face response states
        # ----------------------------

        # Cold start (model loading)
        if response.status_code == 503:
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
                continue
            return {
                "error": "Hugging Face model is loading. Please retry in ~30 seconds."
            }

        # Authentication
        if response.status_code == 401:
            return {
                "error": "Invalid Hugging Face token. Check Streamlit secrets."
            }

        # Rate limit
        if response.status_code == 429:
            return {
                "error": "Hugging Face rate limit exceeded. Please wait and retry."
            }

        # Other errors
        if response.status_code != 200:
            return {
                "error": f"Hugging Face API error {response.status_code}: {response.text}"
            }

        # ----------------------------
        # Success path
        # ----------------------------
        try:
            outputs = response.json()
        except Exception as e:
            return {"error": f"Invalid JSON from Hugging Face: {e}"}

        detections = []

        for obj in outputs:
            box = obj.get("box", {})
            detections.append({
                "label": obj.get("label", "unknown"),
                "confidence": float(obj.get("score", 0.0)),
                "bbox": [
                    box.get("xmin", 0),
                    box.get("ymin", 0),
                    box.get("xmax", 0),
                    box.get("ymax", 0),
                ]
            })

        return detections

    return {"error": "Detection failed after retries."}


# import requests
# import io
# import os

# HF_MODEL = "facebook/detr-resnet-50"
# HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# HEADERS = {
#     "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
# }

# def detect_objects(image):
#     """
#     Returns a list of detections:
#     [
#       {
#         "label": str,
#         "confidence": float,
#         "bbox": [x1, y1, x2, y2]
#       }
#     ]
#     """

#     buf = io.BytesIO()
#     image.save(buf, format="JPEG")
#     buf.seek(0)

#     response = requests.post(
#         HF_API_URL,
#         headers=HEADERS,
#         data=buf.getvalue(),
#         timeout=60
#     )

#     response.raise_for_status()
#     outputs = response.json()

#     detections = []

#     for obj in outputs:
#         box = obj["box"]
#         detections.append({
#             "label": obj["label"],
#             "confidence": obj["score"],
#             "bbox": [
#                 box["xmin"],
#                 box["ymin"],
#                 box["xmax"],
#                 box["ymax"]
#             ]
#         })

#     return detections


# import requests
# import io
# import os

# HF_MODEL = "facebook/detr-resnet-50"
# HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# HEADERS = {
#     "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
# }

# def detect_objects(image):
#     """
#     Returns a list of detections:
#     [
#       {
#         "label": str,
#         "confidence": float,
#         "bbox": [x1, y1, x2, y2]
#       }
#     ]
#     """

#     buf = io.BytesIO()
#     image.save(buf, format="JPEG")
#     buf.seek(0)

#     response = requests.post(
#         HF_API_URL,
#         headers=HEADERS,
#         data=buf.getvalue(),
#         timeout=60
#     )

#     response.raise_for_status()
#     outputs = response.json()

#     detections = []

#     for obj in outputs:
#         box = obj["box"]
#         detections.append({
#             "label": obj["label"],
#             "confidence": obj["score"],
#             "bbox": [
#                 box["xmin"],
#                 box["ymin"],
#                 box["xmax"],
#                 box["ymax"]
#             ]
#         })

#     return detections
