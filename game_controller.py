"""
game_controller.py
---------------------
STEP 4 (part A): Gesture -> game action mapping.

Translates a predicted gesture id into a real keyboard key press so it can
control Subway Surfers (or any keyboard-controlled game/emulator window).

Uses `pyautogui` to send key presses. If your game is a DirectX / fullscreen
exe and pyautogui's key presses don't register, swap pyautogui for
`pydirectinput` (same API: pydirectinput.press("up")) -- install with
`pip install pydirectinput` (Windows only).

IMPORTANT: click on / focus the game window before running so the key
presses are sent to the game, not to PyCharm.
"""

import time
import pyautogui

from config import GESTURES, GESTURE_COOLDOWN_SECONDS

# Map our internal action names (from config.GESTURES) to actual keyboard keys.
# Edit these if your game uses different controls.
ACTION_TO_KEY = {
    "right": "right",
    "left": "left",
    "down": "down",
    "up": "up",
    "powerup": "space",   # change to whatever key your game uses for "power up"
    "none": None,         # closed hand -> do nothing
}


class GameController:
    def __init__(self, cooldown=GESTURE_COOLDOWN_SECONDS):
        self.cooldown = cooldown
        self.last_action_time = 0.0
        self.last_gesture_id = None

    def handle_gesture(self, gesture_id):
        """
        Call this once per frame with the currently predicted gesture id
        (or None if no confident gesture was detected this frame).
        Internally debounces so a gesture held across many frames doesn't
        spam the same key 30 times a second.
        """
        if gesture_id is None:
            self.last_gesture_id = None
            return

        action = GESTURES[gesture_id]["action"]
        key = ACTION_TO_KEY.get(action)

        now = time.time()
        is_new_gesture = gesture_id != self.last_gesture_id
        cooldown_elapsed = (now - self.last_action_time) >= self.cooldown

        if key is not None and (is_new_gesture or cooldown_elapsed):
            pyautogui.press(key)
            self.last_action_time = now
            print(f"-> action: {action}  (key: {key})")

        self.last_gesture_id = gesture_id
