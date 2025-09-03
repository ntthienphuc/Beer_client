#!/usr/bin/env python3
# Offline Beer-Detector – no networking
# © 2025 Phuc Nguyen

import os, sys, time, logging, threading
from pathlib import Path

import cv2
from models.inference import TFLiteModel, UNKNOWN_NAME
import hardware_controller as HW

# ─────────────── GPIO setup
ON_PI = HW.ON_PI
LIGHT_PIN, BUTTON_PIN = 13, 26      # đèn & nút

if ON_PI:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LIGHT_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUTTON_PIN, GPIO.IN,  pull_up_down=GPIO.PUD_UP)
else:                               # giả lập khi chạy desktop
    class _Dummy:
        def __getattr__(self, *_): return lambda *a, **k: None
    GPIO = _Dummy()

# ─────────────── logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OfflineClient")

# ─────────────── model & đếm
MODEL = TFLiteModel(str(Path(__file__).with_name("best.tflite")))
BEER_GROUP = {
    "Bia Heineken": True,  # để phân luồng LED
    "Bia Budweiser": False,
    "Bia Tiger": False,
    "Bia Tiger Bạc": False,
    UNKNOWN_NAME: False,
}
qtys = {k: 0 for k in BEER_GROUP}

def reset_led():
    for k in qtys: qtys[k] = 0
    HW.update_led_displays(qtys)

def add_beer(name: str):
    if name not in qtys: qtys[name] = 0
    qtys[name] += 1
    HW.update_led_displays(qtys)

# ─────────────── image capture / predict
def _predict_img(p: Path):
    img = cv2.imread(str(p))
    return MODEL.predict(img) if img is not None else None

def capture_predict():
    if ON_PI:
        tmp = Path("/tmp/beer.jpg")
        os.system("libcamera-still -o {} --width 640 --height 480 --timeout 500 --nopreview"
                  .format(tmp))
        return _predict_img(tmp)
    # desktop: mở hộp thoại chọn ảnh
    from tkinter import Tk, filedialog
    Tk().withdraw()
    p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
    return _predict_img(Path(p)) if p else None

# ─────────────── nút bấm (reset / shutdown)
class ButtonWatcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
    def run(self):
        while True:
            if GPIO.input(BUTTON_PIN) == 0:             # nhấn ↓
                t_down = time.time()
                while GPIO.input(BUTTON_PIN) == 0:
                    if time.time() - t_down >= 5:
                        logger.warning("Giữ 5 s → shutdown")
                        os.system("sudo shutdown -h now")
                        return
                    time.sleep(0.02)
                # nhấn ngắn → reset
                reset_led()
                logger.info("Nút nhấn: RESET về 00")
                time.sleep(0.3)                          # debounce
            time.sleep(0.05)

# ─────────────── vòng lặp cảm biến
def sensor_loop():
    while True:
        if ON_PI and HW.read_sensor():
            GPIO.output(LIGHT_PIN, GPIO.HIGH)
            beer = capture_predict()
            GPIO.output(LIGHT_PIN, GPIO.LOW)
            if beer:
                add_beer(beer)
                HW.control_servo()
            time.sleep(5)               # debounce + chờ servo
        else:
            time.sleep(0.05)

# ─────────────── main
def main():
    if ON_PI:
        HW.setup_gpio()
    reset_led()
    ButtonWatcher().start()
    logger.info("=== OFFLINE MODE: sẵn sàng nhận diện bia ===")
    try:
        sensor_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if ON_PI:
            HW.cleanup_gpio()

if __name__ == "__main__":
    main()
