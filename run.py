#!/usr/bin/env python
# encoding: utf-8

from control.jumpy_control import Jumpy, JumpyHelper
import cv2
from pynput import mouse
from pynput.mouse import Button
from threading import Thread
import time
import json
import os
import numpy as np
from keras.models import Sequential
import queue


def main():
    folder = "./gameplay_logs"
    with open(f"{folder}/nn_model_config.json", "r") as f:
        config = json.load(f)

    classifier = Sequential.from_config(config)
    classifier.load_weights(f"{folder}/nn_weights.hdf5")
    print(classifier.input_shape[1])
    j = Jumpy()
    recording, draw_positions, draw_prediction = 0, 1, 1
    previous_features = None
    previous_outputs = 0
    i = 0
    prev_x_drags = [0] * previous_outputs
    q = queue.Queue()

    while 1:
        img = j.next_rgb()
        helper = JumpyHelper(img)
        if helper.is_playing_screen():

            if draw_positions:
                positions = helper.get_classified_positions()
                for (x, y, type) in positions:
                    color = (255, 255, 255)
                    if type == "ball":
                        color = (255, 0, 0)
                    if type == "land":
                        color = (0, 255, 0)
                    if type == "dangerous":
                        color = (0, 0, 255)
                    cv2.circle(img, (x, y), 4, color, 2)

            features = np.atleast_2d(helper.get_features())
            to_predict = features
            print("features shape", features.shape)
            if previous_outputs and \
                    classifier.input_shape[1] == to_predict.shape[1] + previous_outputs:
                to_predict = np.append(
                    to_predict, np.atleast_2d(prev_x_drags), axis=1)
            if classifier.input_shape[1] > to_predict.shape[1]:
                if previous_features is None:
                    previous_features = np.zeros_like(features)
                to_predict = np.append(features, previous_features, axis=1)
            y = classifier.predict(to_predict)
            y = round(y[0, 0])
            print(y)

            if draw_prediction:
                img = cv2.putText(img, "{}".format(y),
                                  (50, img.shape[0]-60),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (0, 0, 0), 2, cv2.LINE_AA)

            j.drag_distance(y)
            previous_features = features

            if previous_outputs:
                del prev_x_drags[-1]
                prev_x_drags = [y] + prev_x_drags
            
            if recording:
                q.put(img)

        else:
            previous_features = None
            if previous_outputs:
                prev_x_drags = [0] * previous_outputs
            while not q.empty():
                i += 1
                item = q.get()
                cv2.imwrite("images/gameplay_{}.png".format(i), item)

        if helper.is_continue_screen():
            j.click_no_thanks_button()
        elif helper.is_main_menu_screen():
            j.click_play_single_button()
        elif helper.is_screboard_screen():
            j.click_back_button()

        cv2.imshow('jumpy', img)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            x = helper.get_features()
            print(x)


###################
if __name__ == "__main__":
    main()