import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict, Tuple
import cv2

class ImageBlurring(object):
    def __init__(self, ksize: Tuple[int, int], probability_threshold: float = 0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold
        self.ksize = ksize

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            blurred_image = cv2.blur(src=image, ksize=self.ksize)
            return {'image': blurred_image, 'keypoints': keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"