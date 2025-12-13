# Reserved for future custom drawing / overlays
from PIL import ImageDraw

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)

    for det in detections:
        box = det["bbox"]
        label = det["label"]
        score = det["confidence"]

        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label} {score:.2f}", fill="red")

    return image
