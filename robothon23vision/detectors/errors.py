class DetectionError(Exception):
    pass

class TooManyCirclesDetectedError(DetectionError):
    pass

class NoCircleDetectedError(DetectionError):
    pass

class NoAreaDetectedError(DetectionError):
    pass
