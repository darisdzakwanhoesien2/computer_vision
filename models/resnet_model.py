import torch
from torchvision.models import resnet50, ResNet50_Weights

_model = None
_transform = None
_labels = None

def load_resnet():
    global _model, _transform, _labels

    if _model is None:
        weights = ResNet50_Weights.DEFAULT
        _model = resnet50(weights=weights)
        _model.eval()

        _transform = weights.transforms()
        _labels = weights.meta["categories"]  # ✅ SAFE SOURCE

    return _model, _transform, _labels


def classify_resnet(image, topk=5):
    model, transform, labels = load_resnet()

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
import torch
from torchvision.models import resnet50, ResNet50_Weights

_model = None
_transform = None
_labels = None

def load_resnet():
    global _model, _transform, _labels

    if _model is None:
        weights = ResNet50_Weights.DEFAULT
        _model = resnet50(weights=weights)
        _model.eval()

        _transform = weights.transforms()
        _labels = weights.meta["categories"]  # ✅ SAFE SOURCE

    return _model, _transform, _labels


def classify_resnet(image, topk=5):
    model, transform, labels = load_resnet()

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
