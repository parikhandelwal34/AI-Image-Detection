import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset
from fft import compute_fft

class FFTDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for label, folder in enumerate(["real", "ai"]):
            path = os.path.join(root_dir, folder)
            for file in os.listdir(path):
                self.data.append((os.path.join(path, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (224,224))

        fft = compute_fft(img)
        fft = cv2.resize(fft, (224,224))
        fft = np.stack([fft]*3, axis=-1)

        img = img / 255.0
        fft = fft / 255.0

        combined = np.concatenate([img, fft], axis=2)
        combined = np.transpose(combined, (2,0,1))

        return torch.tensor(combined, dtype=torch.float32), torch.tensor(label)