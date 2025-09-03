# hardware_controller.py
import time, logging, threading
logger = logging.getLogger("Sensor")

_servo_lock = threading.Lock()

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except (ImportError, RuntimeError):
    ON_PI = False

LED_CLOCK_PIN = 25
LED_DATA_PIN  = 24
LED_LATCH_PIN_1 = 8   # Heineken
LED_LATCH_PIN_2 = 7   # Bia khác
SERVO_PIN = 23
SENSOR_PIN = 17

number_codes = [0xC0, 0xF9, 0xA4, 0xB0, 0x99,
                0x92, 0x82, 0xF8, 0x80, 0x90]

# ─────────────── GPIO helpers
def setup_gpio():
    if not ON_PI: return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup([LED_CLOCK_PIN, LED_DATA_PIN,
                LED_LATCH_PIN_1, LED_LATCH_PIN_2, SERVO_PIN], GPIO.OUT)
    GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def cleanup_gpio():
    if ON_PI:
        GPIO.cleanup()

def shift_out(value):
    for i in range(8):
        bit = (value >> (7 - i)) & 1
        GPIO.output(LED_DATA_PIN, bit)
        GPIO.output(LED_CLOCK_PIN, GPIO.HIGH)
        GPIO.output(LED_CLOCK_PIN, GPIO.LOW)

def display_on_module(latch_pin, number):
    if not ON_PI: return
    number = max(0, min(number, 99))
    tens_code = number_codes[number // 10]
    ones_code = number_codes[number % 10]
    GPIO.output(latch_pin, GPIO.LOW)
    shift_out(ones_code)
    shift_out(tens_code)
    GPIO.output(latch_pin, GPIO.HIGH)

def update_led_displays(qtys):
    if not ON_PI: return
    heineken, others = 0, 0
    for name, cnt in qtys.items():
        (heineken if "heineken" in name.lower() else others).__iadd__(cnt)
    display_on_module(LED_LATCH_PIN_1, heineken)
    display_on_module(LED_LATCH_PIN_2, others)
    logger.info("Cập nhật LED: Heineken=%s, Bia khác=%s", heineken, others)

# ─────────────── servo
def control_servo():
    if not ON_PI: return
    with _servo_lock:
        pwm = GPIO.PWM(SERVO_PIN, 50)  # SG90 → 50 Hz
        try:
            pwm.start(2.5)             # đóng
            time.sleep(0.3)
            pwm.ChangeDutyCycle(7.5)   # mở ~90°
            time.sleep(0.5)
            pwm.ChangeDutyCycle(2.5)   # đóng lại
            time.sleep(0.5)
            pwm.ChangeDutyCycle(0)     # ngừng xung
            time.sleep(0.1)
        finally:
            pwm.stop()

def read_sensor() -> bool:
    return ON_PI and GPIO.input(SENSOR_PIN) == GPIO.LOW
