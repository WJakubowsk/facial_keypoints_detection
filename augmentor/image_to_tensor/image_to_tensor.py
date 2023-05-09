import numpy as np
import torch
from typing import Dict

class ImageToTensor(object):

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
        image = np.transpose(image, (2, 0, 1)).copy()

        image = torch.from_numpy(image).type(torch.FloatTensor)
        keypoints = torch.from_numpy(keypoints).type(torch.FloatTensor)

        return {'image': image, 'keypoints': keypoints}