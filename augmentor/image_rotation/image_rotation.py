import numpy as np
from torchvision.utils import _log_api_usage_once
from typing import Dict, Tuple
import cv2
import math

class ImageRotation(object):
    def __init__(self, probability_threshold: float = 0.5, angle: float = 30):
        super().__init__()
        _log_api_usage_once(self)
        self.probability_threshold = probability_threshold
        self.angle = angle

    def _rotate_point(self, origin: Tuple[float, float], point: Tuple[float, float], angle: float) -> Tuple[float, float]:
        xo, yo = origin
        xp, yp = point

        x_final = xo + math.cos(math.radians(angle)) * (xp - xo) - math.sin(math.radians(angle)) * (yp - yo)
        y_final = yo + math.sin(math.radians(angle)) * (xp - xo) + math.cos(math.radians(angle)) * (yp - yo)
        return x_final, y_final

    def _rotate_image(self, image: np.array, angle: float) -> np.array:
        image_center_coordinates = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center_coordinates, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image

    def __call__(self, image_with_keypoints: Dict[np.array, np.array]) -> Dict[np.array, np.array]:
        if np.random.random() < self.probability_threshold:
            image, keypoints = image_with_keypoints["image"], image_with_keypoints["keypoints"]
            # image rotation
            rotated_image = self._rotate_image(image, -self.angle)
            
            # keypoints rotation
            image_width, image_height = image.shape[0], image.shape[1]
            origin = (image_width / 2, image_height / 2)
            rotated_keypoints = np.copy(keypoints)
            for i, point in enumerate(keypoints.reshape(15, 2)):
                new_keypoint = self._rotate_point(origin, point, self.angle)
                rotated_keypoints[i * 2] = new_keypoint[0]
                rotated_keypoints[i * 2 + 1] = new_keypoint[1]
        
            return {'image': rotated_image, 'keypoints': rotated_keypoints}

        return image_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability_threshold})"