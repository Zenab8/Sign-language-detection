# WARNING: pyautogui runs actions on the machine hosting this Python process.
# Do NOT enable this on a cloud server unless you intend to control that server's OS.
import logging
try:
    import pyautogui
except Exception:
    pyautogui = None
    logging.warning("pyautogui not available; local actions will not run.")

def perform_action(gesture_label: str):
    # Simple example: control slides or press keys
    try:
        if pyautogui is None:
            return
        if gesture_label == "ThumbsUp":
            pyautogui.press("right")
        elif gesture_label == "ThumbsDown":
            pyautogui.press("left")
        elif gesture_label == "Fist":
            pyautogui.press("space")
        elif gesture_label == "Palm":
            # move mouse to center (example)
            screen_width, screen_height = pyautogui.size()
            pyautogui.moveTo(screen_width/2, screen_height/2)
    except Exception as e:
        logging.exception("perform_action failed: %s", e)
