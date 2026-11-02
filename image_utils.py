from __future__ import annotations
import cv2
import numpy as np

def safe_get_camera_frame(cam) -> tuple[np.ndarray | None, bool]:
    """Retrieves live frame array from Basler PylonCamera object using stock grab()."""
    if cam is None:
        return None, False

    try:
        # PylonCamera stock method is grab()
        if hasattr(cam, 'grab'):
            raw = cam.grab()
        elif hasattr(cam, 'get_image'):
            raw = cam.get_image()
        elif hasattr(cam, 'get_frame'):
            raw = cam.get_frame()
        else:
            return None, False

        if raw is None:
            return None, False

        frame = np.array(raw, copy=True)
        if frame.size == 0:
            return None, False

        return frame, True
    except Exception as e:
        print(f"[image_utils] Error getting camera frame: {e}")
        return None, False

def safe_set_led(leds, channel_name: str, state: bool, power_ma: int = 100):
    """Controls stock LEDBoard channels ('BF', 'PC', 'DF', 'Red', 'Blue', 'Green')."""
    if leds is None:
        return

    channel_map = {
        "Brightfield": "BF",
        "Phase": "PC",
        "Darkfield": "DF",
        "Red": "Red",
        "Blue": "Blue",
        "Green": "Green"
    }
    ch_code = channel_map.get(channel_name, "BF")

    try:
        if not state:
            if hasattr(leds, 'led_off'):
                leds.led_off(ch_code)
            elif hasattr(leds, 'leds_off'):
                leds.leds_off()
        else:
            if hasattr(leds, 'led_on'):
                leds.led_on(ch_code, power_ma)
            elif hasattr(leds, 'leds_enable'):
                leds.leds_enable()
    except Exception as e:
        print(f"[image_utils] Error setting LED channel {channel_name}: {e}")

def generate_disconnected_camera_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates a dark placeholder frame with MICROSCOPE NOT CONNECTED overlay."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "MICROSCOPE NOT CONNECTED",
        (width // 6, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    return img

def generate_imaging_error_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generates an error placeholder frame with IMAGING ERROR overlay."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "IMAGING ERROR",
        (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 165, 255),
        2,
        cv2.LINE_AA
    )
    return img
