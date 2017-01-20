import argparse
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from exercise.sliding_window import slide_window


def load_scaler():
    # load the scaler
    scaler_file_name = "x_scaler.pkl"
    scaler_load = joblib.load(scaler_file_name)
    return scaler_load


def load_classifier_model():
    # load the classifier model
    model_file_name = "linearSVC_model.pkl"
    svc_load = joblib.load(model_file_name)
    return svc_load


def detect_vehicles_image_name(img_name):
    img = mpimg.imread(img_name)
    return detect_vehicles_image(img)


def detect_vehicles_image(input_img,
                          scaler=load_scaler(),
                          classifier=load_classifier_model(),
                          view=True):
    img = np.copy(input_img)
    xy_window=(64, 64)
    img_height = img.shape[0]
    y_start_stop=[img_height/2, img_height]


    window_list = slide_window(
        img, xy_window=xy_window, y_start_stop=y_start_stop)
    if view:
        for i, window in enumerate(window_list):
            if i < xy_window[0]*xy_window[1]:
                (startx,starty), (endx, endy)= window
                plt.subplot(xy_window[0], xy_window[1], i+1, xticks=[], yticks=[])
                plt.imshow(img[startx:endx, starty:endy, :])
        plt.show()

    return img


def detect_vehicles_video(video_name):
    pass


## Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        help='Train the classifier')
    parser.add_argument(
        '--image',
        default="test/frame981.jpg",
        help='image to be processed')
    parser.add_argument(
        '--video',
        default="project_video.mp4")
    parser.add_argument(
        '--visual',
        action='store_true',
        default=False)
    args = parser.parse_args()
    view = None
    if args.visual:
        view = None
    if args.train:
        pass
    if args.image is not None:
        detect_vehicles_image_name(args.image)
        exit()
    if args.video is not None:
        detect_vehicles_video(args.video)
