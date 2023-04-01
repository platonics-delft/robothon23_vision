class FeatureDetector(object):
    def __init__(self, img_in, name: str):
        self._img = img_in.copy()
        self._name = name
        self._filtered_img = None

    def img(self):
        return self._img

    def low(self):
        return [
            self._config.h_low, 
            self._config.s_low, 
            self._config.v_low, 
            ]

    def high(self):
        return [
            self._config.h_high, 
            self._config.s_high, 
            self._config.v_high, 
            ]

    def filtered_img(self):
        return self._filtered_img

    def detect(self) -> dict:
        pass

