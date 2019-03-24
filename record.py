#!/usr/bin/env python3
# encoding: utf-8

from control.jumpy_control import Jumpy, JumpyHelper
import cv2
from pynput import mouse
from pynput.mouse import Button
from threading import Thread
import time
import json
import os


class MouseListener():

    def __init__(self):
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click)
        self.dragging = False
        self.playing = False
        self.jumpy = Jumpy()
        self.drag_start_x = None
        self.drag_start_y = None
        self.logs = []
        self.last_sent_position = (None, None)
        self.current_position = (None, None)

    def start_watching(self):
        self.listener.start()
        # self.listener.join()

    def on_click(self, x, y, button, pressed):
        if self.playing and button == Button.left:
            if pressed and not self.dragging:
                self.dragging = True
                # print("clicked", x, y)
                self.log_click(x, y)
            elif not pressed and self.dragging:
                self.dragging = False
                # print("unclicked")
                self.log_unclick(x, y)
            # print("mouse click",x,y,button,pressed,sep=" || ")

    def in_monitor(self, x, y):
        monitor = self.jumpy.settings['monitor']
        if x < monitor['left']:
            return False
        if x > monitor['left'] + monitor['width']:
            return False
        if y < monitor['top']:
            return False
        if y > monitor['top'] + monitor['height']:
            return False
        return True

    def on_move(self, x, y):
        if self.playing and self.dragging and self.in_monitor(x, y):
            # print("dragging", x, y, sep=" || ")
            self.log_drag(x, y)

    def log_drag(self, x, y):
        self.current_position = (x, y)
        self.logs.append({
            "type": "drag",
            "x": x,
            "y": y,
            "time": time.time()
        })

    def log_click(self, x, y):
        self.current_position = self.last_sent_position = (x, y)
        self.logs.append({
            "type": "click",
            "x": x,
            "y": y,
            "time": time.time()
        })

    def log_unclick(self, x, y):
        self.current_position = self.last_sent_position = (None, None)
        self.logs.append({
            "type": "unclick",
            "x": x,
            "y": y,
            "time": time.time()
        })

    def clear_logs(self):
        self.logs = []

    def save_logs(self, filename):
        if self.logs:
            with open(filename, "w") as f:
                json.dump(self.logs, f)

    def pop_last_drag_amount(self):
        if self.last_sent_position != (None, None) and \
                self.current_position != (None, None):
            to_return = [pos[0]-pos[1]
                         for pos in zip(self.current_position, self.last_sent_position)]
        else:
            to_return = (0, 0)

        self.last_sent_position = self.current_position
        return to_return


def record():
    mouse_listener = MouseListener()
    mouse_listener.start_watching()
    previous_screen = -1
    is_logging = False
    root_path = "gameplay_logs"
    full_path = None
    game_info = None
    i = 0
    while 1:
        curr_time = time.time()
        img = mouse_listener.jumpy.get_monitor_rgb()
        helper = JumpyHelper(img)
        if helper.is_playing_screen():
            mouse_listener.playing = True
        else:
            mouse_listener.playing = False

        if previous_screen != helper.UNKNOWN and \
                helper.current_screen() != previous_screen and \
                helper.is_playing_screen():
            print("game just started!")

            # if not os.path.exists(root_path):
            # os.makedirs(root_path)
            full_path = root_path+"/"+str(curr_time)
            if not os.path.exists(full_path+"/images"):
                os.makedirs(full_path+"/images")

            mouse_listener.clear_logs()
            i = 0
            game_info = {
                "start": curr_time,
                "images": {

                }
            }

            is_logging = True

        elif is_logging and not helper.is_unknown_screen() and \
                helper.current_screen() != previous_screen and \
                previous_screen == helper.PLAYING_SCREEN:
            print("game just eneded!")
            score = helper.get_score_integer()
            game_info["end"] = curr_time
            game_info["score"] = score

            with open(full_path+"/game_info.json", "w") as f:
                json.dump(game_info, f)
            mouse_listener.save_logs(full_path+"/mouse_logs.json")
            is_logging = False

        if is_logging:
            i += 1
            image_name = "{}.png".format(i)
            drag_x, drag_y = mouse_listener.pop_last_drag_amount()
            
            game_info["images"][image_name] = {
                "time": curr_time,
                "prev_drag_x": drag_x,
                "prev_drag_y": drag_y
            }
            cv2.imwrite(full_path+"/images/"+image_name, img)            
            img = cv2.putText(img, "{},{}".format(drag_x, drag_y),
                              (50, img.shape[0]-40),
                              cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
        previous_screen = helper.current_screen()
        cv2.imshow('jumpy', img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break

    eventsLogger.stop()


###################
if __name__ == "__main__":
    record()
