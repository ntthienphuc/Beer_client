# models/inference.py
import os, cv2, numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

THRESHOLD    = 0.6
UNKNOWN_NAME = "Món Chưa Xác Định"
IMG_SIZE     = 224
CLASS_NAMES  = ["budweiser", "heniken", "tiger", "tiger_bac"]
NAME_MAPPING = {
    "budweiser": "Bia Budweiser",
    "heniken":   "Bia Heineken",
    "tiger":     "Bia Tiger",
    "tiger_bac": "Bia Tiger Bạc",
}

class TFLiteModel:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        self.interp = Interpreter(model_path)
        self.interp.allocate_tensors()
        self.inp_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]

    # ─────────────── preprocessing
    def _prep(self, img):
        h, w = img.shape[:2]
        r = IMG_SIZE / max(h, w)
        nh, nw = int(h * r), int(w * r)
        img = cv2.resize(img, (nw, nh))
        ph, pw = IMG_SIZE - nh, IMG_SIZE - nw
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img[None]

    # ─────────────── predict
    def predict(self, img, thr: float = THRESHOLD) -> str:
        self.interp.set_tensor(self.inp_idx, self._prep(img))
        self.interp.invoke()
        probs = self.interp.get_tensor(self.out_idx)[0]
        idx = int(np.argmax(probs))
        if probs[idx] < thr:
            return UNKNOWN_NAME
        cls = CLASS_NAMES[idx]
        return NAME_MAPPING.get(cls, cls)
