import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

_model = None
_transform = None
_labels = None

def load_efficientnet():
    global _model, _transform, _labels

    if _model is None:
        weights = EfficientNet_B0_Weights.DEFAULT
        _model = efficientnet_b0(weights=weights)
        _model.eval()

        _transform = weights.transforms()
        _labels = weights.meta["categories"]  # ✅ SAFE SOURCE

    return _model, _transform, _labels


def classify_efficientnet(image, topk=5):
    model, transform, labels = load_efficientnet()

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits[0], dim=0)

    values, indices = torch.topk(probs, topk)

    results = []
    for idx, score in zip(indices.tolist(), values.tolist()):
        results.append({
            "label": labels[idx],
            "confidence": float(score)
        })

    return results
