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
    "right": "d",
    "left": "a",
    "down": "down",
    "up": "space",
    "powerup": None,   # change to whatever key your game uses for "power up"
    "none": None,         # closed hand -> do nothing
}


class GameController:
    def __init__(self, cooldown=GESTURE_COOLDOWN_SECONDS):
        self.cooldown = cooldown
        self.last_action_time = 0.0
        self.last_gesture_id = None
        self.last_key = None

    def handle_gesture(self, gesture_id):

        # No gesture → release held key
        if gesture_id is None:
            if self.last_key:
                pyautogui.keyUp(self.last_key)
                self.last_key = None

            self.last_gesture_id = None
            return

        action = GESTURES[gesture_id]["action"]
        key = ACTION_TO_KEY.get(action)

        # Release previous key if changing gestures
        if key != self.last_key:

            if self.last_key:
                pyautogui.keyUp(self.last_key)

            if key:
                pyautogui.keyDown(key)

            self.last_key = key

        self.last_gesture_id = gesture_id
