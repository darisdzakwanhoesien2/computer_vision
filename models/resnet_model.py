import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

_model = None
_transform = None

def load_resnet():
    global _model, _transform
    if _model is None:
        weights = ResNet50_Weights.DEFAULT
        _model = resnet50(weights=weights)
        _model.eval()
        _transform = weights.transforms()
    return _model, _transform

def classify_resnet(image, topk=5):
    model, transform = load_resnet()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs[0], dim=0)

    values, indices = torch.topk(probs, topk)
    return indices.tolist(), values.tolist()