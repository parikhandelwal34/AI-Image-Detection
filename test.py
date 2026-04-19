import torch
import cv2
import numpy as np
from model import get_model
from fft import compute_fft

model = get_model()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))

    fft = compute_fft(img)
    fft = cv2.resize(fft, (224, 224))
    fft = np.stack([fft]*3, axis=-1)

    img = img / 255.0
    fft = fft / 255.0

    combined = np.concatenate([img, fft], axis=2)
    combined = np.transpose(combined, (2, 0, 1))

    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)

    output = model(tensor)
    _, pred = torch.max(output, 1)

    return "AI Image" if pred.item() == 1 else "Real Image"

print(predict("test.jpg"))