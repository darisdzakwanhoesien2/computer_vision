import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

_model = None
_transform = None

def load_efficientnet():
    global _model, _transform
    if _model is None:
        weights = EfficientNet_B0_Weights.DEFAULT
        _model = efficientnet_b0(weights=weights)
        _model.eval()
        _transform = weights.transforms()
    return _model, _transform

def classify_efficientnet(image, topk=5):
    model, transform = load_efficientnet()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs[0], dim=0)

    values, indices = torch.topk(probs, topk)
    return indices.tolist(), values.tolist()