from dataclasses import dataclass
import cv2
import numpy as np

from robothon23.detectors.feature_detector import FeatureDetector
from robothon23.detectors.errors import NoCircleDetectedError, TooManyCirclesDetectedError


@dataclass
class TemplateConfig:
    min_radius: float = 11
    max_radius: float = 15
    circle_param_1: float = 50
    circle_param_2: float = 9
    kernel: bool = False
    h_low: int = 100
    s_low: int = 140
    v_low: int = 150
    h_up: int = 115
    s_up: int = 255
    v_up: int = 225

class TemplateDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'blue_button')
        self._config = TemplateDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        # Detect circles using HoughCircles

        self._mask = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low, self._config.s_low, self._config.v_low]),
                np.array([self._config.h_up, self._config.s_up, self._config.v_up]),
                )
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
            raise NoCircleDetectedError("No blue button detected.")
        else:
            if not circles.shape == (1, 1, 3):
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    cv2.circle(self._filtered_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                raise TooManyCirclesDetectedError("Too many blue circles detected.")
            # Draw the outer circle
            blue_circle = (
                    int(circles[:, 0, 0]),
                    int(circles[:, 0, 1]),
                    int(circles[:, 0, 2])
                    )
            cv2.circle(
                    self._img,
                    blue_circle[0:2],
                    blue_circle[2],
                     (0, 255, 0),
                     2)
            cv2.circle(self._img, blue_circle[0:2], 2, (0, 0, 255), 3)

        results = {'template': template[0:2]}
        return results

