import numpy as np
from robothon23vision.detectors.blue_area_detector import BlueAreaDetector
from robothon23vision.detectors.blue_button_detector import BlueButtonDetector
from robothon23vision.detectors.errors import DetectionError
from robothon23vision.detectors.gray_area_detector import GrayAreaDetector
from robothon23vision.detectors.red_button_detector import RedButtonDetector
from robothon23vision.utils.compute_tf import compute_transform

class Localizer(object):

    def __init__(self, img):
        self._detected_points = {}
        self._reference_points = {}
        self._img = img
        self._ground_truth = {
                'gray_area': (329, 168),
                'red_button': (521, 158),
                'blue_button': (522, 138),
                'blue_area': (523, 215),
        }
        self._pixel_cm_factor = 320.0/0.25
        self._points = {}
        self._detectors = []
        self._detectors.append(RedButtonDetector(self._img))
        self._detectors.append(BlueButtonDetector(self._img))
        self._detectors.append(GrayAreaDetector(self._img))
        self._detectors.append(BlueAreaDetector(self._img))

        for detector in self._detectors:
            detector.set_mask()


    def detect_points(self):
        for detector in self._detectors:
            try:
                res = detector.detect()
                self._points.update(res)
            except DetectionError as e:
                print(e)

    def red_image(self):
        return self._red_img

    def blue_image(self):
        return self._blue_img

    def points(self) -> np.ndarray:
        values = list(self._points.values())
        return np.transpose(np.array(values))

    def ground_truth_points(self) -> np.ndarray:
        values = list(self._ground_truth.values())
        return np.transpose(np.array(values))


    def compute_tf(self) -> np.ndarray:
        p0 = []
        p1 = []
        for key, value in self._points.items():
            p0.append(self._ground_truth[key])
            p1.append(value)
        p0 = np.transpose(np.array(p0))
        p1 = np.transpose(np.array(p1))
        T0 = compute_transform(p0, p1)
        #T0[0:2, 2] /= self._pixel_cm_factor
        return T0

    def compute_full_tf_in_m(self) -> np.ndarray:
        T0 = self.compute_tf()
        T0[0:2, 2] /= self._pixel_cm_factor
        T_gio = np.identity(4)
        T_gio[0:2, 0:2] = T0[0:2, 0:2]
        T_gio[0:2, 3] = T0[0:2, 2]
        return T_gio

