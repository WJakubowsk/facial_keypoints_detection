import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict

class ImageFlip(object):
    def __init__(self, probability_threshold: float = 0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            # image flip
            flipped_image = np.fliplr(image)

            # keypoints flip
            image_width = image.shape[0]
            reorder = [2, 3, 0, 1, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 21, 24, 25, 22, 23, 26, 27, 28, 29] # reorder of the keypoints to obtain symmetrical positions
            flipped_keypoints = np.array([keypoints[i] for i in reorder])
            
            # apply additional transformations
            for i in range(0, 29, 2):
                flipped_keypoints[i] = image_width - flipped_keypoints[i]

            return {'image': flipped_image, 'keypoints': flipped_keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"