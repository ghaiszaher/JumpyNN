# JumpyNN

This project is a trial to train a neural network to solve [Facebook's Jumpy Jumpy Game](https://www.facebook.com/instantgames/198982457542294)

## Dependencies

Make sure that the following are installed:
* Python3
* Keras
* Tensorflow
* Opencv2 & Numpy
* Pillow
* pynput
* pyautogui
* mss

## Adjust Configuration Files
For this version, auto-calibration was not made. You need to open [Facebook's Jumpy Jumpy Game](https://www.facebook.com/instantgames/198982457542294) and go to the file "jumpy_settings" to adjust the following:
* monitor.left & monitor.top: the location of the game's div relative to the screen
* monitor.width & monitor.height: the size of the game's div

## Capture Data
Run the file "record.py" and play the game to collect images & outputs for your gameplay.
A new folder will be created "gameplay_logs" that contains images + mouse dragging movement info during the gameplay.

## Train
TODO

## Test Trained Data
TODO