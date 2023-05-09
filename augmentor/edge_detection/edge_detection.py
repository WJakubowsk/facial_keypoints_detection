import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict
import cv2

class EdgeDetection(object):
    def __init__(self, probability_threshold: float = 0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            kernel = np.array([[-1, -1, -1], [-1, 8,-1], [-1, -1, -1]])
            sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            return {'image': sharpened_image, 'keypoints': keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"