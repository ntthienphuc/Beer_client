#models/inference.py
import os
import cv2
import numpy as np
try:
    # On Raspberry Pi we ship the tiny wheel
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback to full TF Lite if you run the same code on a PC
    from tensorflow.lite.python.interpreter import Interpreter

THRESHOLD     = 0.8                      # ngưỡng tự tin tối thiểu
UNKNOWN_NAME  = "Món Chưa Xác Định"       # phải khớp menu.csv
IMG_SIZE = 224
CLASS_NAMES = ["budweiser", "heniken", "tiger", "tiger_bac"]
NAME_MAPPING = {
    "budweiser": "Bia Budweiser",
    "heniken":   "Bia Heineken",
    "tiger":     "Bia Tiger",
    "tiger_bac": "Bia Tiger Bạc",
}

class TFLiteModel:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.interp = Interpreter(model_path)
        self.interp.allocate_tensors()
        self.input_idx  = self.interp.get_input_details()[0]["index"]
        self.output_idx = self.interp.get_output_details()[0]["index"]

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        r = IMG_SIZE / max(h, w)
        nh, nw = int(h*r), int(w*r)
        img = cv2.resize(img_bgr, (nw, nh))
        pad_w, pad_h = IMG_SIZE - nw, IMG_SIZE - nh
        top, bottom = pad_h//2, pad_h - pad_h//2
        left, right = pad_w//2, pad_w - pad_w//2
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=[114,114,114])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        return np.expand_dims(img, axis=0)

    def predict(self, img_bgr: np.ndarray, thr: float = THRESHOLD) -> str:
        """
        Trả về tên món theo NAME_MAPPING; nếu max prob < thr → UNKNOWN_NAME
        """
        inp = self._preprocess(img_bgr)
        self.interp.set_tensor(self.input_idx, inp)
        self.interp.invoke()
        probs = self.interp.get_tensor(self.output_idx)[0]

        best_idx  = int(np.argmax(probs))
        best_prob = probs[best_idx]

        if best_prob < thr:
            return UNKNOWN_NAME               # ← tự gán món chưa xác định
        cls = CLASS_NAMES[best_idx]
        return NAME_MAPPING.get(cls, cls)
