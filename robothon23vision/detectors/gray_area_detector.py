from dataclasses import dataclass
import cv2
import numpy as np

from robothon23vision.detectors.feature_detector import FeatureDetector
from robothon23vision.detectors.errors import NoAreaDetectedError


@dataclass
class GrayAreaDetectorConfig:
    kernel: bool = False
    min_area: int = 3000
    h_low: int = 90
    s_low: int = 30
    v_low: int = 150
    h_high: int = 110
    s_high: int = 100
    v_high: int = 175

class GrayAreaDetector(FeatureDetector):
    def __init__(self, img_in, **config):
        super().__init__(img_in, 'gray_area')
        self._config = GrayAreaDetectorConfig(**config)

    def set_mask(self) -> None:
        self._hsv = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)

        # Detect circles using HoughCircles

        self._mask = cv2.inRange(
                self._hsv,
                np.array([self._config.h_low, self._config.s_low, self._config.v_low]),
                np.array([self._config.h_high, self._config.s_high, self._config.v_high]),
                )
        if self._config.kernel:
            kernel = np.ones((5,5), np.uint8)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, kernel)
            self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, kernel)



    def detect(self) -> dict:
        gray_img = cv2.bitwise_and(self._hsv, self._hsv, mask=self._mask)

        self._filtered_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(self._filtered_img, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(self._mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        centers = []
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self._config.min_area:
                areas.append(area)
                max_area = area
                max_contour = contour
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append(np.array([cX, cY]))
                # draw the contour and center of the shape on the image
                cv2.drawContours(self._img, [contour], -1, (0, 255, 0), 2)
                #cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

        if len(centers) == 0:
            raise NoAreaDetectedError("No gray area found.")

        centers = np.array(centers)
        weights = np.array(areas)
        true_center_x = np.average(centers[:, 0],weights=weights)
        true_center_y = np.average(centers[:, 1],weights=weights)
        cx = int(true_center_x)
        cy = int(true_center_y)
        cv2.circle(self._img, (cx, cy) , 5, (0, 255, 0), -1)
        result = {self._name: (cx, cy)}
        return result


