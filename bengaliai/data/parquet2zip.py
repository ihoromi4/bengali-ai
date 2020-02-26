import os
import zipfile
import gc
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from .crop_resize import crop_resize

HEIGHT = 137
WIDTH = 236


def parquet_to_images(filenames: str, out_filename: str, size: int = 128):
    if os.path.exists(out_filename):
        print('File', out_filename, 'already exists. Abort.')
        return

    print('Create file:', out_filename)
    for filename in filenames:
        df = pd.read_parquet(filename)

        with zipfile.ZipFile(out_filename, 'a') as f_zip:
            print('Parquet loaded.')

            data = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
            data = 255 - data
            
            names = list(df.iloc[:, 0])

            gc.collect()

            print('Start processing images...')
            for idx in tqdm(range(len(df))):
                name = names[idx]

                img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
                img = crop_resize(img, size)

                img = cv2.imencode('.png', img)[1]
                f_zip.writestr(name + '.png', img)

