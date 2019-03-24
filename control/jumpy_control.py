#!/usr/bin/env python
# encoding: utf-8

import mss
import cv2
import numpy as np
from PIL import Image
import json
import pyautogui
import random
import math
import time
from pynput import keyboard


class Jumpy:

    settings_path = 'settings/jumpy_settings.json'

    default_settings = {
        # full screen
        # "monitor": {
        #     "left": 482,
        #     "top": 20,
        #     "width": 387,
        #     "height": 728
        # },

        # chrome normal mode
        "monitor": {
            "left": 519,
            "top": 150,
            "width": 360,
            "height": 597
        },
        "drag_duration": 0,
        "distance_pi_ratio": 167/(2*math.pi),
        "buttons": {
            "back": {
                "rel_y": 19/360,
                "rel_x": 20/597
            },
            "play_single": {
                "rel_y": 0.5,
                "rel_x": 467/597
            },
            "no_thanks": {
                "rel_y": 0.5,
                "rel_x": 543/597
            }
        }
    }

    def __init__(self):
        self.load_settings()
        self.capture = Capture(monitor=self.settings['monitor'])
        pyautogui.PAUSE = 0

    def load_settings(self):
        try:
            with open(self.settings_path, 'r') as f:
                self.settings = json.load(f)
        except IOError:
            self.settings = self.default_settings

            with open(self.settings_path, 'w') as file:
                json.dump(self.settings, file, indent=4)

    def get_monitor_rgb(self):
        return self.capture.monitor_rgb(monitor=self.settings['monitor'])

    def next_rgb(self):
        return self.get_monitor_rgb()

    def get_screen_rgb(self):
        return self.capture.screen_rgb()

    def center_mouse(self, click=True):
        x = self.settings['monitor']['left'] + \
            self.settings['monitor']['width']//2
        y = self.settings['monitor']['top'] + \
            self.settings['monitor']['height']-20  # //2
        pyautogui.moveTo(x, y)
        if click:
            pyautogui.click()

    def drag_distance(self, distance):
        self.center_mouse()
        pyautogui.dragRel(
            distance, duration=self.settings['drag_duration'], _pause=False)

    def drag_angle(self, angle):
        distance = angle * self.settings['distance_pi_ratio']
        self.drag_distance(distance)

    # x,y are relative to the monitor (jumpy cropped image)
    def click(self, x, y):
        x += self.settings['monitor']['left']
        y += self.settings['monitor']['top']
        pyautogui.click(x, y)
        # pyautogui.moveTo(x, y)

    def get_position_from_rel(self, rel_x, rel_y):
        return self.settings['monitor']['width']*rel_y, self.settings['monitor']['height']*rel_x

    # rel_x, rel_y are the proportional position related to monitor
    def click_rel(self, rel_x, rel_y):
        x, y = self.get_position_from_rel(rel_x, rel_y)
        self.click(x, y)

    def get_button_rel_position(self, button_key):
        rel_x = self.settings['buttons'][button_key]['rel_x']
        rel_y = self.settings['buttons'][button_key]['rel_y']

        return rel_x, rel_y

    def click_button(self, button_key):
        rel_x = self.settings['buttons'][button_key]['rel_x']
        rel_y = self.settings['buttons'][button_key]['rel_y']
        self.click_rel(rel_x, rel_y)

    def click_back_button(self):
        self.click_button('back')

    def click_play_single_button(self):
        self.click_button('play_single')

    def click_no_thanks_button(self):
        self.click_button('no_thanks')


class Capture:
    def __init__(self, monitor=None):
        self.sct = mss.mss()
        self.screen_width = self.sct.monitors[0]["width"]
        self.screen_height = self.sct.monitors[0]["height"]
        self.all_screen_monitor = {
            'top': 0,
            'left': 0,
            'width': self.screen_width,
            'height': self.screen_height
        }

        self.monitor = monitor if monitor else self.all_screen_monitor

    def rgb(self, monitor=None):
        img = self.sct.grab(monitor if monitor else self.monitor)
        img = Image.frombytes('RGB', img.size, img.rgb)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def monitor_rgb(self, monitor=None):
        return self.rgb(monitor)

    def screen_rgb(self):
        return self.rgb(self.all_screen_monitor)


class JumpyHelper:

    UNKNOWN = -1
    MAIN_MENU_SCREEN = 1
    CONTINUE_SCREEN = 2
    SCOREBOARD_SCREEN = 3
    PLAYING_SCREEN = 4

    _current_screen = None
    _ball_mask = None
    _safe_land_mask = None
    _dangerous_land_mask = None
    _space_mask = None
    _white_mask = None
    _hsv = None
    _gray = None

    default_settings = {
        "hsv_masks": {
            "dangerous_land": {
                "low": [36, 10, 0],
                "high": [70, 255, 255]
            },
            "safe_land": {
                "low": [90, 229, 0],
                "high": [110, 255, 255]
            },
            "ball": {
                "low": [135, 10, 0],
                "high": [150, 255, 255]
            },
            "white": {
                "low": [0, 0, 229],
                "high": [180, 20, 255]
            }
        },
        "templates": {
            "continue": {
                "path": "templates/template_continue.png"
            },
            "score_board": {
                "path": "templates/template_score_board.png"
            },
            "main_menu": {
                "path": "templates/template_main_menu.png"
            }
        },

        "areas": {
            "ball": {
                "rel_x": 72/597,
                "rel_y": 166/360,
                "rel_height": 128/597,
                "rel_width": 26/360,
                "min_mean": 0.1
            },
            "score": {
                "playing": {
                    "min_rel_x": 15/597,
                    "max_rel_x": 64/597
                },
                "continue": {
                    "min_rel_x": 75/597,
                    "max_rel_x": 125/597
                },
                "scoreboard": {
                    "min_rel_x": 64/597,
                    "max_rel_x": 123/597,
                    "max_rel_y": 280/360
                }
            }
        }
    }

    settings = None
    templates = None
    features_positions = None

    settings_path = 'settings/jumpy_helper_settings.json'
    features_positions_path = 'settings/features_positions_map.json'

    def __init__(self, rgb):
        JumpyHelper.__load_settings__()
        JumpyHelper.__load_templates__()
        JumpyHelper.__load_features_positions__()
        self.img = rgb

    @classmethod
    def __load_templates__(cls):
        if cls.templates:
            return

        cls.templates = {}
        if "templates" not in cls.settings:
            return

        for t in cls.settings["templates"]:
            cls.templates[t] = cls.settings["templates"][t]
            cls.templates[t]["image"] = cv2.imread(
                cls.templates[t]["path"], 0)
            cls.templates[t]["height"] = cls.templates[t]["image"].shape[0]
            cls.templates[t]["width"] = cls.templates[t]["image"].shape[1]

    @classmethod
    def __load_settings__(cls):
        if cls.settings:
            return

        # print("Loading Settings...")
        try:
            with open(cls.settings_path, 'r') as f:
                cls.settings = json.load(f)
        except IOError:
            cls.settings = cls.default_settings
            with open(cls.settings_path, 'w') as file:
                json.dump(cls.settings, file, indent=4)

    @classmethod
    def __load_features_positions__(cls):
        if cls.features_positions:
            return

        try:
            with open(cls.features_positions_path, 'r') as f:
                cls.features_positions = json.load(f)
        except IOError:
            pass

    def get_features(self):
        """
        Returns a python list of zeros and ones, each pixel will be represented by
        3 consecutive values:
            * 1 0 0 if the pixel is ball
            * 0 1 0 if the pixel is safe area
            * 0 0 1 if the pixel is dangerous area
            * 0 0 0 otherwise
        """

        if self.features_positions is None:
            return None

        X = []

        for position in self.features_positions:
            if type(position) is dict:
                x, y = position["x"], position["y"]
            else:
                x, y = position

            if self.get_ball_mask()[y, x]:
                # X += [0]
                X += [1, 0, 0]
            elif self.get_safe_land_mask()[y, x]:
                # X += [0]
                X += [0, 1, 0]
            elif self.get_dangerous_land_mask()[y, x]:
                # X += [-1]
                X += [0, 0, 1]
            else:
                # X += [1]
                X += [0, 0, 0]

        return X

    def get_classified_positions(self):
        if self.features_positions is None:
            return None

        X = []

        for position in self.features_positions:
            if type(position) is dict:
                x, y = position["x"], position["y"]
            else:
                x, y = position[0], position[1]

            if self.get_ball_mask()[y, x]:
                X.append((x, y, "ball"))
            elif self.get_safe_land_mask()[y, x]:
                X.append((x, y, "land"))
            elif self.get_dangerous_land_mask()[y, x]:
                X.append((x, y, "dangerous"))
            else:
                X.append((x, y, "none"))

        return X

    def find_template(self, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.8, min_point=None, max_point=None):
        """
        Returns 3 values;
            * Boolean: if the template was found or not
            * max_val: evaluation of the algorithm
            * top_left: of the found template
        """

        img = self.get_gray()
        if min_point or max_point:
            img = img.copy()
            w = img.shape[1]
            h = img.shape[0]
            if not min_point:
                min_point = (0, 0)
            if not max_point:
                max_point = (h, w)

            img = img[min_point[0]:max_point[0], min_point[1]:max_point[1]]

        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val <= threshold:
            return False, max_val, (-1, -1)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        if min_point:
            top_left = (top_left[1] + min_point[0], top_left[1] + min_point[1])

        return True, max_val, top_left

    def get_cropped_ball_area_mean(self):
        mask = self.get_ball_mask()
        rel_x = self.settings['areas']['ball']['rel_x']
        rel_y = self.settings['areas']['ball']['rel_y']
        rel_width = self.settings['areas']['ball']['rel_width']
        rel_height = self.settings['areas']['ball']['rel_height']
        x = int(rel_x * mask.shape[0])
        y = int(rel_y * mask.shape[1])
        height = int(rel_height * mask.shape[0])
        width = int(rel_width * mask.shape[1])
        top_left = (x, y)
        bottom_right = (x+height, y+width)
        # print(top_left, bottom_right)
        cropped = mask[top_left[0]:bottom_right[0],
                       top_left[1]:bottom_right[1]]
        ratio = cropped[cropped > 0].size / cropped.size
        # cv2.imshow("cropped", cropped)
        return ratio

    def current_screen(self):
        if self._current_screen:
            return self._current_screen
        # Do calculations:
        templates = self.templates
        found, max_val, top_left = self.find_template(
            templates["continue"]["image"], min_point=(self.img.shape[0]//3, 0), max_point=(2*self.img.shape[0]//3, self.img.shape[1]))
        if found:
            self._current_screen = self.CONTINUE_SCREEN
            return self._current_screen

        score_board_max_size = round(self.img.shape[0]*50/597)
        found, max_val, top_left = self.find_template(
            templates["score_board"]["image"], max_point=(score_board_max_size, score_board_max_size))
        if found:
            self._current_screen = self.SCOREBOARD_SCREEN
            return self._current_screen

        found, max_val, top_left = self.find_template(
            templates["main_menu"]["image"], min_point=(self.img.shape[0]//2, 0))
        if found:
            self._current_screen = self.MAIN_MENU_SCREEN
            return self._current_screen
        if self.get_cropped_ball_area_mean() > self.settings['areas']['ball']['min_mean']:
            self._current_screen = self.PLAYING_SCREEN
            return self._current_screen

        self._current_screen = self.UNKNOWN
        return self._current_screen

    def is_main_menu_screen(self):
        return self.current_screen() == self.MAIN_MENU_SCREEN

    def is_continue_screen(self):
        return self.current_screen() == self.CONTINUE_SCREEN

    def is_screboard_screen(self):
        return self.current_screen() == self.SCOREBOARD_SCREEN

    def is_playing_screen(self):
        return self.current_screen() == self.PLAYING_SCREEN

    def is_unknown_screen(self):
        return self.current_screen() == self.UNKNOWN

    def get_hsv(self):
        if self._hsv is not None:
            return self._hsv
        self._hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        return self._hsv

    def get_gray(self):
        if self._gray is not None:
            return self._gray
        self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    def get_mask(self, low, high):
        return cv2.inRange(self.get_hsv(), tuple(low),
                           tuple(high))

    def get_masked_rgb_from_mask(self, mask):
        imask = mask > 0
        masked = np.zeros_like(self.img, np.uint8)
        masked[imask] = self.img[imask]
        return masked

    def get_masked_rgb(self, low, high):
        mask = self.get_mask(low, high)
        return self.get_masked_rgb_from_mask(mask)

    def get_ball_mask(self):
        if self._ball_mask is not None:
            return self._ball_mask
        self._ball_mask = self.get_mask(self.settings['hsv_masks']['ball']['low'],
                                        self.settings['hsv_masks']['ball']['high'])
        return self._ball_mask

    def get_safe_land_mask(self):
        if self._safe_land_mask is not None:
            return self._safe_land_mask
        self._safe_land_mask = self.get_mask(self.settings['hsv_masks']['safe_land']['low'],
                                             self.settings['hsv_masks']['safe_land']['high'])
        return self._safe_land_mask

    def get_dangerous_land_mask(self):
        if self._dangerous_land_mask is not None:
            return self._dangerous_land_mask
        self._dangerous_land_mask = self.get_mask(self.settings['hsv_masks']['dangerous_land']['low'],
                                                  self.settings['hsv_masks']['dangerous_land']['high'])
        return self._dangerous_land_mask

    def get_white_mask(self):
        if self._white_mask is not None:
            return self._white_mask
        self._white_mask = self.get_mask(self.settings['hsv_masks']['white']['low'],
                                         self.settings['hsv_masks']['white']['high'])
        return self._white_mask

    def get_space_mask(self):
        if self._space_mask is not None:
            return self._space_mask
        imask = (self.get_ball_mask() == 0) & (
            self.get_dangerous_land_mask() == 0) & (self.get_safe_land_mask() == 0)
        self._space_mask = np.zeros_like(self.get_ball_mask(), np.uint8)
        self._space_mask[imask] = 255
        return self._space_mask

    def get_ball_masked(self):
        return self.get_masked_rgb_from_mask(self.get_ball_mask())

    def get_dangerous_land_masked(self):
        return self.get_masked_rgb_from_mask(self.get_dangerous_land_mask())

    def get_safe_land_masked(self):
        return self.get_masked_rgb_from_mask(self.get_safe_land_mask())

    def get_white_masked(self):
        return self.get_masked_rgb_from_mask(self.get_white_mask())

    def get_space_masked(self):
        return self.get_masked_rgb_from_mask(self.get_space_mask())

    def ocr_white(self, min_x=None, min_y=None, max_x=None, max_y=None):
        white = self.get_white_masked()
        black = cv2.bitwise_not(white)
        if not min_x:
            min_x = 0
        if not min_y:
            min_y = 0
        if not max_x:
            max_x = black.shape[0]
        if not max_y:
            max_y = black.shape[1]
        cropped = black[min_x:max_x, min_y:max_y]

        img = Image.fromarray(cropped)
        img.info['dpi'] = (96, 96)
        # text = pytesseract.image_to_string(img)
        text = "UNKNOWN"
        return text

    def ocr_white_key(self, screen_key):
        min_x = None
        max_x = None
        min_y = None
        max_y = None

        if "min_rel_x" in self.settings["areas"]["score"][screen_key]:
            min_rel_x = self.settings["areas"]["score"][screen_key]["min_rel_x"]
            min_x = int(min_rel_x * self.img.shape[0])
        if "max_rel_x" in self.settings["areas"]["score"][screen_key]:
            max_rel_x = self.settings["areas"]["score"][screen_key]["max_rel_x"]
            max_x = int(max_rel_x * self.img.shape[0])
        if "min_rel_y" in self.settings["areas"]["score"][screen_key]:
            min_rel_y = self.settings["areas"]["score"][screen_key]["min_rel_y"]
            min_y = int(min_rel_y * self.img.shape[1])
        if "max_rel_y" in self.settings["areas"]["score"][screen_key]:
            max_rel_y = self.settings["areas"]["score"][screen_key]["max_rel_y"]
            max_y = int(max_rel_y * self.img.shape[1])

        return self.ocr_white(min_x, min_y, max_x, max_y)

    def get_score_string(self):
        if self.is_playing_screen():
            # return self.ocr_white(min_x=15, max_x=64)
            return self.ocr_white_key("playing")

        if self.is_continue_screen():
            return self.ocr_white(min_x=75, max_x=125)
            return self.ocr_white_key("continue")

        if self.is_screboard_screen():
            return self.ocr_white(min_x=64, max_x=123, max_y=280)
            return self.ocr_white_key("scoreboard")

        return ""

    def get_score_integer(self):
        score = self.get_score_string()
        try:
            return int(score)
        except ValueError:
            score = ''.join(c for c in score if c.isdigit())
            if score:
                return int(score)

        return 0


if __name__ == "__main__":
    '''
    This code only tests the module's functionaities
    '''
    button = None
    is_keyboard = False
    keyboard_thread = None
    j = None
    ctrl = False

    j = Jumpy()
    j.center_mouse()
    orx, ory = pyautogui.position()
    # print(orx, ory)
    quit = False
    start = 0
    maxstart = 500
    i = 0
    previous_screen = -1
    previous_screen_clicked = -1
    display_score = "-"
    last_score_read = time.time()
    showing = {'jumpy': True}
    while 1:
        i += 1
        if quit:
            cv2.destroyAllWindows()
            break
        # img = c.all_screen_rgb()
        img = j.get_monitor_rgb()
        original_img = img.copy()
        helper = JumpyHelper(original_img)
        x, y = pyautogui.position()
        x = orx-x
        y = ory-y

        t = time.time()
        sc = helper.current_screen()
        # print("Time needed to find screen: ", time.time()-t)
        # print("Screen is:", sc)

        font_color = (255, 255, 255)
        if helper.is_continue_screen():
            screen = "Continue Screen"
        elif helper.is_main_menu_screen():
            screen = "Main Menu Screen"
        elif helper.is_screboard_screen():
            screen = "Scoreboard Screen"
        elif helper.is_playing_screen():
            screen = "Playing Screen"
        else:
            screen = "Unknown"
            font_color = (0, 0, 255)
        # print(screen)

        # mean = helper.get_cropped_ball_area_mean()
        # print(mean)
        # if not helper.is_playing_screen() and start:
        # start = 0
        score = "-"
        if (helper.is_continue_screen() or helper.is_screboard_screen()) and previous_screen != helper.current_screen():
            score = helper.get_score_integer()
            last_score_read = time.time()

        if helper.is_playing_screen() and (last_score_read is None or time.time()-last_score_read > 10):
            score = helper.get_score_integer()
            last_score_read = time.time()

        display_score = score if score != "-" else display_score
        img = cv2.putText(img, "{}".format(display_score),
                          (50, img.shape[0]-60),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, "{}".format(screen),
                          (50, img.shape[0]-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, "{}".format(screen),
                          (50, img.shape[0]-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, font_color, 1, cv2.LINE_AA)
        # img = cv2.putText(img, "{}".format(mean),
        #                   (50, img.shape[0]-40),
        #                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        # img = cv2.putText(img, "{},{}".format(x, y), (200, 40),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        # img = cv2.putText(img, "t: {}".format(time.time()), (50, 10),
        #   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        # img = j.get_screen_rgb()

        def show_window(title, image):
            if title in showing and showing[title]:
                cv2.imshow(title, image)
            else:
                cv2.destroyWindow(title)

        show_window('jumpy', img)
        show_window('dangerous', helper.get_dangerous_land_mask())
        show_window('safe', helper.get_safe_land_mask())
        show_window('ball', helper.get_ball_mask())
        show_window('space', helper.get_space_mask())
        show_window('white', helper.get_white_mask())

        # Uncomment if you want to save the images
        # cv2.imwrite("images/img_{}.png".format(i), original_img)
        # cv2.imwrite("images/img_ball_{}.png".format(i), helper.get_ball_mask())
        # cv2.imwrite("images/img_safe_{}.png".format(i), helper.get_safe_land_mask())
        # cv2.imwrite("images/img_dangerous_{}.png".format(i), helper.get_dangerous_land_mask())
        # cv2.imwrite("images/img_ball_masked_{}.png".format(i), helper.get_ball_masked())
        # cv2.imwrite("images/img_safe_masked_{}.png".format(i), helper.get_safe_land_masked())
        # cv2.imwrite("images/img_dangerous_masked_{}.png".format(i), helper.get_dangerous_land_masked())
        if start:
            if True or helper.current_screen() != previous_screen_clicked:
                if helper.is_continue_screen():
                    j.click_no_thanks_button()
                    previous_screen_clicked = helper.current_screen()
                elif helper.is_main_menu_screen():
                    j.click_play_single_button()
                    previous_screen_clicked = helper.current_screen()
                elif helper.is_screboard_screen():
                    j.click_back_button()
                    previous_screen_clicked = helper.current_screen()
            if helper.is_playing_screen():
                dist = np.random.randint(-167//2, 167//2)
                low = -math.pi
                high = math.pi
                # angle = np.random.rand()*(high-low)+low
                choices = np.random.normal(0, math.pi/4, 1000)
                choices = choices[choices < math.pi]
                # choices = [low] * 5 + [high]
                angle = random.choice(choices)
                # angle = high
                move = np.random.randint(0, 4)
                if True:  # move==3:
                    # dist = 167//2
                    # j.drag_distance(dist)
                    j.drag_angle(angle)
                # j.drag(dist)
            start -= 1
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            quit = True
            break

        #########################################
        #########################################
        ####### TEMPORARY *** KEYBOARD #########
        #########################################
        #########################################
        keyboard_angle = math.pi/12

        def on_press(key):
            global button
            global ctrl
            try:
                print('alphanumeric key {0} pressed'.format(
                    key.char))
                button = key.char
            except AttributeError:
                print('special key {0} pressed'.format(
                    key))
            if key == keyboard.Key.left:
                button = 'left'
                # j.drag_angle(keyboard_angle)
            elif key == keyboard.Key.right:
                button = 'right'
                # j.drag_angle(-keyboard_angle)
            elif key == keyboard.Key.ctrl:
                print("ctrl=True")
                ctrl = True
            elif key == keyboard.Key.esc:
                # Stop listener
                global is_keyboard
                is_keyboard = False
                return False

        def on_release(key):
            global ctrl
            global button

            if key == keyboard.Key.ctrl:
                print("ctrl=False")
                ctrl = False
            elif key in (keyboard.Key.left, keyboard.Key.right):
                button = None
            print('{0} released'.format(
                key))

        if k & 0xFF == ord('k'):
            is_keyboard = not is_keyboard
        if is_keyboard:
            if not keyboard_thread or not keyboard_thread.is_alive():
                keyboard_thread = keyboard.Listener(
                    on_press=on_press, on_release=on_release)
                keyboard_thread.start()

            angle = keyboard_angle
            if ctrl:
                angle *= 2
            if button == 'left':
                j.drag_angle(-angle)
            elif button == 'right':
                j.drag_angle(angle)
        elif button:
            button = None

        #########################################
        #########################################
        #########################################
        #########################################
        if k & 0xFF == ord('s'):
            start = maxstart if not start else 0
            j.center_mouse()
            # pyautogui.click()
        # if maxstart-start==10:#k & 0xFF == ord('r'):
            # pyautogui.click()
            # pyautogui.press('f5')
        if k & 0xFF == ord('m'):
            j.center_mouse()

        key_show = {
            '1': 'jumpy',
            '2': 'dangerous',
            '3': 'safe',
            '4':  'ball',
            '5':  'space',
            '6':  'white'
        }
        if k & 0xFF >= ord('1') and k & 0xFF <= ord('6'):
            window_name = key_show[chr(k & 0xFF)]
            if window_name in showing and showing[window_name]:
                showing[window_name] = False
            else:
                showing[window_name] = True

        previous_screen = helper.current_screen()

