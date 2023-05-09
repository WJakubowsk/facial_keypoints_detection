import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transforms=None):
        self.df = dataframe
        self.transforms = transforms
        
        self.images = []
        self.keypoints = []
        
        for _, row in self.df.iterrows():
            # Image
            image = row['Image']
            image = np.stack((image, image, image), axis=-1) # treating image as RGB (with 3 channels instead of 1)

            # Keypoints
            keypoints = row.drop(['Image'])
            keypoints = keypoints.to_numpy().astype('float32')
            
            # Add to Dataset's images and keypoints
            self.images.append(image)
            self.keypoints.append(keypoints)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        keypoints = self.keypoints[idx]
        
        return_dict = {'image': image, 'keypoints': keypoints}

        if self.transforms:
            return_dict = self.transforms(return_dict)

        return return_dict

    def __len__(self):
        return len(self.df)