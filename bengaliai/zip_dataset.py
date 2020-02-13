import io
import zipfile
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset


class ZIPImageDataset(Dataset):
    def __init__(self, filename: str, labels: pd.DataFrame, columns: str, transform = None, cache_mem: bool = True):
        if cache_mem:
            with open(filename, 'rb') as f:
                filename = io.BytesIO(f.read())
                
        self.file = zipfile.ZipFile(filename, 'r')
        self.names = self.file.namelist() if (labels is None) else list(labels['image_id'] + '.png')
        self.labels = labels
        self.columns = columns
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        
        buffer = np.frombuffer(self.file.read(name), dtype='uint8')
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        if self.labels is None:
            output = {n:0 for n in self.columns}
        else:
            output = self.labels.iloc[idx][self.columns].to_dict()
            
            
        output['images'] = image
        
        return output

