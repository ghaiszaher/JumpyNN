# JumpyNN

This project is a trial to train a neural network to play [Facebook's Jumpy Jumpy Game](https://www.facebook.com/instantgames/198982457542294)

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
* Jupyter Notebook

## Adjust Configuration Files
For this version, auto-calibration was not made. You need to open [Facebook's Jumpy Jumpy Game](https://www.facebook.com/instantgames/198982457542294) and go to the file "jumpy_settings" to adjust the following:
* monitor.left & monitor.top: the location of the game's div relative to the screen
* monitor.width & monitor.height: the size of the game's div

## Adjust Template Files
TODO

## Capture Data
Run the file "record.py" and play the game to collect images & outputs for your gameplay.
A new folder will be created "gameplay_logs" that contains images + mouse dragging movement info during the gameplay.

## Train
* Extract features from images and generate Training Data: Run the jupyter notebook 's file generate_X_y. This will generate several files called "features_X_y.npz", the final one will be in "gameplay_logs/features_X_y.npz", and will contain the training data from all the images combined.
* Run the jupyter notebook's file "train.ipynb" to create and run the neural network. Final Neural Network files (config & weights) will be stored in "gameplay_logs"

## Test Trained Data
Run the file "run.py" to test

## Results
TODO

## Note
If you're trying to run this on Windows, you might have some issues with mouse drag using pyautogui library.

## TODO: Add Images

## TODO: Add Results