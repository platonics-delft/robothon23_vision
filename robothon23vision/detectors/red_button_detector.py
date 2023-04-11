from dataclasses import dataclass
import cv2
import numpy as np

from robothon23vision.detectors.feature_detector import FeatureDetector
from robothon23vision.detectors.errors import NoCircleDetectedError, TooManyCirclesDetectedError


@dataclass
class RedButtonDetectorConfig:
    min_radius: float = 7
    max_radius: float = 10
    circle_param_1: float = 20
    circle_param_2: float = 5
    kernel: bool = False
    h_low: int = 0
    s_low: int = 80
    v_low: int = 150
    h_up: int = 10
    s_up: int = 200
    v_up: int = 255

class RedButtonDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'red_button')
        self._config = RedButtonDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        mask_0 = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low, self._config.s_low, self._config.v_low]),
                np.array([self._config.h_up, self._config.s_up, self._config.v_up]),
                )
        self._mask = mask_0

        if self._config.kernel:
            kernel = np.ones((5,5), np.uint8)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, kernel)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, kernel)



    def detect(self) -> dict:
        circles = cv2.HoughCircles(
                self._mask,
                cv2.HOUGH_GRADIENT,
                1,
                20,
                param1=self._config.circle_param_1,
                param2=self._config.circle_param_2,
                minRadius=self._config.min_radius,
                maxRadius=self._config.max_radius,
                )

        self._filtered_img = cv2.bitwise_and(self._img, self._img, mask=self._mask)

        if circles is None:
            raise NoCircleDetectedError("No red circle detected.")
        else:
            circles = np.uint16(np.around(circles))
            # Selecting logic if there are too circles use highest y coordinate
            circle = circles[0, np.argmin(circles[:, :, 1]), :]

            red_button = (
                    int(circle[0]),
                    int(circle[1]),
                    int(circle[2])
                    )
            cv2.circle(
                    self._img,
                    red_button[0:2],
                    red_button[2],
                     (0, 255, 255),
                     2)
            cv2.circle(self._img, red_button[0:2], 2, (0, 255, 255), 3)

            results = {
                    self._name:  red_button[0:2],
                    }
            return results

