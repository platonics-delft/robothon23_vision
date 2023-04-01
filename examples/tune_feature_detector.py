import time
import sys
import cv2

import argparse
from pathlib import Path


from robothon23vision.detectors.blue_button_detector import BlueButtonDetector
from robothon23vision.detectors.red_circles_detector import RedCirclesDetector
from robothon23vision.detectors.silver_circle_detector import SilverCircleDetector
from robothon23vision.detectors.gray_area_detector import GrayAreaDetector
from robothon23vision.detectors.errors import *

detector_map = {
        'blue_button': BlueButtonDetector,
        'red_circles': RedCirclesDetector,
        'silver_circle': SilverCircleDetector,
        'gray_area': GrayAreaDetector,
        }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--detector_name", "-d", type=str,default='blue_button')
    parser.add_argument("--picture", "-p", type=str,default="data/localization_img_1_rgb.png")


    args = parser.parse_args()
    img_name = args.picture
    detector_name = args.detector_name
    img_in = cv2.imread(img_name)
    cv2.imshow('image', img_in)
    cv2.waitKey(0)

    t0 = time.perf_counter()
    detector = detector_map[detector_name](img_in)
    low = detector.low()
    high = detector.high()
    while True:
        print(f"low : {low}")
        print(f"high: {high}")
        try:
            input_low = input("Give low value\n")
            if len(input_low) > 1:
                low = [int(value) for value in input_low.split(' ')]
            input_high = input("Give high value\n")
            if len(input_high) > 1:
                high = [int(value) for value in input_high.split(' ')]
        except Exception as e:
            print(e)
        detector = detector_map[detector_name](
                img_in,
                h_low=low[0],
                s_low=low[1],
                v_low=low[2],
                h_high=high[0],
                s_high=high[0],
                v_high=high[0],
                )
        detector.set_mask()
        try:
            res = detector.detect()
            cv2.imshow('image', detector.filtered_img())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("No button detected")
            cv2.imshow('image', detector.filtered_img())
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
