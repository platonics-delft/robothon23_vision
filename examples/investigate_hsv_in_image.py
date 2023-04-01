import sys

from robothon23vision.utils.get_color_under_cursor import get_color_under_cursor


def main():
    img_name = sys.argv[1]
    get_color_under_cursor(img_name)


if __name__ == "__main__":
    main()
