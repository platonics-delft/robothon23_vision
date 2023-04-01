import time
import sys
import argparse
import cv2

from robothon23vision.detectors.blue_button_detector import BlueButtonDetector
from robothon23vision.detectors.red_circles_detector import RedCirclesDetector
from robothon23vision.detectors.silver_circle_detector import SilverCircleDetector
from robothon23vision.detectors.gray_area_detector import GrayAreaDetector
from robothon23vision.detectors.red_button_detector import RedButtonDetector
from robothon23vision.detectors.blue_area_detector import BlueAreaDetector
from robothon23vision.detectors.errors import *

detector_map = {
        'blue_button': BlueButtonDetector,
        'red_circles': RedCirclesDetector,
        'silver_circle': SilverCircleDetector,
        'gray_area': GrayAreaDetector,
        'red_button': RedButtonDetector,
        'blue_area': BlueAreaDetector,
        }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector_name", "-d", type=str,default='blue_button')
    parser.add_argument("--picture", "-p", type=str,default="data/localization_img_1_rgb.png")


    args = parser.parse_args()
    img_name = args.picture
    detector_name = args.detector_name

    img_in = cv2.imread(img_name)
    detector = detector_map[detector_name](img_in)
    detector.set_mask()
    try:
        res = detector.detect()
        print(res)
        cv2.imshow('image', detector.img())
    except DetectionError as e:
        print(e)
        cv2.imshow('image', detector.filtered_img())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
